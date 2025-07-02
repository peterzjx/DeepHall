import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
import numpy as np
from deephall import dmc_sample, Config
from deephall.config import Network, NetworkType, System, PsiformerNetwork,Network, NetworkType, FluxType, FermionicType, PartonNetwork
from pathlib import Path
from deephall.types import LogPsiNetwork, get_walker_state
import jax
import jax.numpy as jnp
import kfac_jax


import pytest
from omegaconf import OmegaConf
from pytest import CaptureFixture

import logging
import jax.numpy as jnp
import time



def run_dmc(simple_config: Config):
    # TODO: load from pretrained vmc checkpoint
    # TODO: calculate initial local energy and logpsi and velocity
    

    # --- Set up logging ---
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(simple_config.log.save_path + '/status_loggings.log')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
# ==================================================================


    log_manager = dmc_sample.LogManager(simple_config)
    model = dmc_sample.make_network(simple_config.system, simple_config.network)
    network = dmc_sample.cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmove = dmc_sample.setup_mcmc(simple_config, network)
    print('initial setup_mcmc done', pmap_mcmc_step)
    _, state = (
        dmc_sample.initalize_state(simple_config, model)
    )
    walker_state = get_walker_state(state)
    params = state.params
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(simple_config.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = jnp.ones(1000)*simple_config.initial_energy
    
    start = time.time()
    for step in range(simple_config.mcmc.burn_in):
        print("Step burn in ", step)
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        walker_state, pmove = pmap_mcmc_step(params, walker_state, subkey)
        
        local_mean_energy = dmc_sample.weighted_mean_energy(walker_state=walker_state)
        energy_history.at[step % len(energy_history)].set(local_mean_energy)
        walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=5000, use_external_energy=True, external_energy=local_mean_energy)
    walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=1, reweight_interval=1, use_external_energy=True, external_energy=local_mean_energy)            
    end = time.time()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds")

    # with log_manager.create_writer() as writer:
    #     renormal_interval = 100
    #     energy_update_interval = 1000
    #     for step in range(simple_config.mcmc.iteration):
    #         print("Step ", step)
    #         sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
    #         walker_state, pmove, acceptance_threhold, accepted_idx, old_walker, xy_move, move, log_green_function_forward, log_green_function_backward  = pmap_mcmc_step(params, walker_state, subkey)
    #         energy_history, mean_energy = dmc_sample.accumulate_energy(walker_state, energy_history, max_length=10000)
    #         walker_state, changed, idx_min, conditioned, change_shape = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=energy_update_interval,reweight_interval=renormal_interval,use_external_energy=True, external_energy=mean_energy)
    #         writer.log(
    #             step=str(step),
    #             pmove=f"{pmove[0]:.2f}",
    #             local_energy=f"{dmc_sample.weighted_mean_energy(walker_state):.6f}",
    #             dmc_mean_energy=f"{jnp.mean(walker_state.dmc_mean_energy):.6f}",
    #             history_mean_energy=f"{mean_energy:.6f}",
    #             weight_max=f"{jnp.max(walker_state.weights):.6f}",
    #             weight_min=f"{jnp.min(walker_state.weights):.6f}",
    #             weight_std=f"{jnp.std(walker_state.weights):.6f}"
    #         )
    #         if step%renormal_interval==0 and (jnp.min(walker_state.weights)<0.01 or jnp.max(walker_state.weights)>5.0) and renormal_interval>10:
    #             renormal_interval = renormal_interval - 1
    #         assert renormal_interval>10
if __name__=="__main__":
    config = Config(network=Network(
            type=NetworkType.laughlin
            # type=NetworkType.parton,
            # parton=PartonNetwork(
            #     fermionic_type=FermionicType.pfaffian,
            #     flux_type=FluxType.symmetric_mlp_network
            # )
        ))
    config.seed = 126
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.0001
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    # config.optim.iterations = 20000
    config.batch_size = 64
    config.mcmc.width = 0.3
    config.initial_energy = config.system.nspins[0] * 0.5 + 0.467 * config.system.nspins[0] * config.system.interaction_strength

    # config.log.pretrained_path = "../logs/pfaf_4_kappa_1.0/ckpt_009978.npz"
    # config.log.save_path = "../logs/pfaf_4_kappa_1.0_dmc"
    # config.log.pretrained_path = "../logs/psiformer_4_kappa_1.0/ckpt_000519.npz"
    # config.log.save_path = "../logs/psiformer_4_kappa_1.0_dmc"
    # config.log.pretrained_path = "../logs/laughlin_4_kappa_1.0/ckpt_003884.npz"
    config.log.save_path = f"../logs/laughlin_4_kappa_{config.system.interaction_strength}_dmc"

    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 100
    config.mcmc.iteration = 50000

    run_dmc(config)