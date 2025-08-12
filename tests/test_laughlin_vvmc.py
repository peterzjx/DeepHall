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

def thetaphi_xy(electron_thetaphi: jnp.ndarray):
    theta = electron_thetaphi[..., 0]
    phi = electron_thetaphi[..., 1]
    x = jnp.cos(phi) / jnp.tan(theta / 2)
    y = jnp.sin(phi) / jnp.tan(theta / 2)
    electron_xy = jnp.stack([x, y], axis=-1)
    return electron_xy


@pytest.fixture
def simple_cfg():
    config = Config()
    config.network.type = NetworkType.laughlin_v
    config.seed = 1
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.interaction_strength = 1.0
    config.optim.iterations = 100
    config.batch_size = 48
    config.mcmc.burn_in = 100
    config.mcmc.iteration = 100
    config.initial_energy = config.system.nspins[0] * 0.5 + 0.467 * config.system.nspins[0] * config.system.interaction_strength
    config.log.initial_energy = False
    config.log.save_path = "../logs/test_logs"
    return config

def test_vvmc(simple_cfg: Config):
    # --- Set up logging ---
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)

    # # File handler
    # file_handler = logging.FileHandler(simple_cfg.log.save_path + '/status_loggings.log')
    # file_handler.setLevel(logging.INFO)

    # # Console handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)

    # # Formatter
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # # Add handlers to the logger
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
# ==================================================================

    
    log_manager = vvmc_sample.LogManager(simple_cfg)
    model = vvmc_sample.make_v_network(simple_cfg.system, simple_cfg.network)
    network = vvmc_sample.cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmove = vvmc_sample.setup_mcmc(simple_cfg, network)
    print('initial setup_mcmc done', pmap_mcmc_step)
    _, state = (
        vvmc_sample.initalize_state(simple_cfg, model)
    )
    walker_state = get_walker_state(state)
    params = state.params

    # print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(simple_cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = jnp.ones(1000)*simple_cfg.initial_energy
    
    start = time.time()

    
    for step in range(simple_cfg.mcmc.burn_in):
        print("Step burn in #", step)
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        walker_state, pmove, acceptance_threshold = pmap_mcmc_step(params, walker_state, subkey)
        # print('p = ',pmove, acceptance_threshold)
        local_mean_energy = vvmc_sample.weighted_mean_energy(walker_state=walker_state)
        energy_history.at[step % len(energy_history)].set(local_mean_energy)
        walker_state = vvmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=5000, use_external_energy=True, external_energy=local_mean_energy)
    walker_state = vvmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=1, reweight_interval=1, use_external_energy=True, external_energy=local_mean_energy)            

    
    with log_manager.create_writer() as writer:
        renormal_interval = 100
        energy_update_interval = 1000
        
        for step in range(simple_cfg.mcmc.iteration):
            
        
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threshold  = pmap_mcmc_step(params, walker_state, subkey)
            # Ensure all device work is finished before continuing
            # walker_state = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, walker_state)
            
            local_mean_energy = vvmc_sample.weighted_mean_energy(walker_state=walker_state)     
            print("Step ", step)
            print('E = ', local_mean_energy, ' p = ', jnp.mean(pmove))  
            energy_history.at[step % len(energy_history)].set(local_mean_energy)
            hist_mean_energy = jnp.mean(energy_history)
            
            walker_state = vvmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=energy_update_interval,reweight_interval=renormal_interval,use_external_energy=True, external_energy=local_mean_energy)
        
            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                local_energy=f"{local_mean_energy:.6f}",
                dmc_mean_energy=f"{jnp.mean(walker_state.dmc_mean_energy):.6f}",
                history_mean_energy=f"{hist_mean_energy:.6f}",
                weight_max=f"{jnp.max(walker_state.weights):.6f}",
                weight_min=f"{jnp.min(walker_state.weights):.6f}",
                weight_std=f"{jnp.std(walker_state.weights):.6f}" 
            )
            
            if step%renormal_interval==0 and (jnp.min(walker_state.weights)<0.01 or jnp.max(walker_state.weights)>5.0) and renormal_interval>10:
                renormal_interval = renormal_interval - 1
            assert renormal_interval>=10