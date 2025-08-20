from pathlib import Path
import jax
import jax.numpy as jnp
import logging
import time
from deephall import Config, vvmc_sample
from deephall.types import LogPsiNetwork, get_walker_state
from deephall.velocity_networks import LaughlinVelocity, SuperLaughlinVelocity
from deephall.config import NetworkType
from deephall.hamiltonian import local_v_energy
import kfac_jax

import pytest
from pytest import CaptureFixture

# def thetaphi_xy(electron_thetaphi: jnp.ndarray):
#     theta = electron_thetaphi[..., 0]
#     phi = electron_thetaphi[..., 1]
#     x = jnp.cos(phi) / jnp.tan(theta / 2)
#     y = jnp.sin(phi) / jnp.tan(theta / 2)
#     electron_xy = jnp.stack([x, y], axis=-1)
#     return electron_xy





def run_vvmc(cfg: Config):
    log_manager = vvmc_sample.LogManager(cfg)
    model = vvmc_sample.make_v_network(cfg.system, cfg.network)
    network = vvmc_sample.cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmove = vvmc_sample.setup_mcmc(cfg, network)
    print('initial setup_mcmc done', pmap_mcmc_step)
    _, state = (
        vvmc_sample.initalize_state(cfg, model)
    )
    walker_state = get_walker_state(state)
    params = state.params

    # print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = jnp.ones(1000)*cfg.initial_energy

    
    for step in range(cfg.mcmc.burn_in):
        print("Step burn in #", step)
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        walker_state, pmove, acceptance_threshold = pmap_mcmc_step(params, walker_state, subkey)
        # print('p = ',pmove, acceptance_threshold)
        local_mean_energy = jnp.mean(walker_state.local_energy)   
        energy_history.at[step % len(energy_history)].set(local_mean_energy)

    
    with log_manager.create_writer() as writer:
        renormal_interval = 100
        energy_update_interval = 1000
        
        for step in range(cfg.mcmc.iteration):
            
        
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threshold  = pmap_mcmc_step(params, walker_state, subkey)
            
            # local_mean_energy = vvmc_sample.weighted_mean_energy(walker_state=walker_state)     
            local_mean_energy = jnp.mean(walker_state.local_energy)     
            print("Step ", step)
            print('E = ', local_mean_energy, ' p = ', jnp.mean(pmove))  
            # print(walker_state.local_energy)
            energy_history.at[step % len(energy_history)].set(local_mean_energy)        
            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                local_energy=f"{local_mean_energy:.6f}",
            )
            


if __name__ == "__main__":
    
    config = Config()
    config.network.type = NetworkType.laughlin_v
    config.seed = 1
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.interaction_strength = 1.0
    config.optim.iterations = 100
    config.batch_size = 1024
    config.mcmc.burn_in = 500
    config.mcmc.iteration = 5000
    config.initial_energy = config.system.nspins[0] * 0.5 + 0.467 * config.system.nspins[0] * config.system.interaction_strength
    config.log.initial_energy = False
    # config.log.pretrained_path = "../logs/super_laughlin_v_fit/ckpt_004999.npz"
    config.log.save_path = "../logs/test_logs"

    run_vvmc(config)
