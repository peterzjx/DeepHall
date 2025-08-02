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

# Add JAX profiler imports
from jax.profiler import trace, TraceAnnotation

import pytest
from omegaconf import OmegaConf
from pytest import CaptureFixture

import logging
import jax.numpy as jnp
import time

def count_param_bytes(params):
    """Recursively count total parameter size in bytes."""
    total = 0
    def _count(pytree):
        nonlocal total
        if isinstance(pytree, dict):
            for v in pytree.values():
                _count(v)
        elif isinstance(pytree, (list, tuple)):
            for v in pytree:
                _count(v)
        elif isinstance(pytree, jax.Array):
            total += pytree.size * pytree.dtype.itemsize
        elif isinstance(pytree, np.ndarray):
            total += pytree.size * pytree.dtype.itemsize
    _count(params)
    return total

def run_vdmc(simple_config: Config):
    # TODO: load from pretrained vmc checkpoint
    # TODO: calculate initial local energy and logpsi and velocity
    

    # --- Set up logging ---
    with TraceAnnotation("logging_setup"):
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

    with TraceAnnotation("setup_initialization"):
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
        total_bytes = count_param_bytes(params)
        print(f"Model size: {total_bytes / (1024**2):.2f} MB")
        # print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
        key = jax.random.PRNGKey(simple_config.seed)
        sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
        energy_history = jnp.ones(1000)*simple_config.initial_energy
    
    start = time.time()
###########################################################
    # def burnin_body(step, carry):
    #     sharded_key, walker_state, energy_history = carry
    #     sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
    #     walker_state, pmove = pmap_mcmc_step(params, walker_state, subkey)
    #     local_mean_energy = dmc_sample.weighted_mean_energy(walker_state=walker_state)
    #     energy_history = energy_history.at[step % energy_history.shape[0]].set(local_mean_energy)
    #     walker_state = dmc_sample.update_mean_energy(
    #         walker_state=walker_state,
    #         step=step,
    #         update_interval=5000,
    #         use_external_energy=True,
    #         external_energy=local_mean_energy
    #     )
    #     return (sharded_key, walker_state, energy_history)

    # # Prepare initial carry
    # init_carry = (sharded_key, walker_state, energy_history)

    # # Run burn-in loop
    # sharded_key, walker_state, energy_history = jax.lax.fori_loop(
    #     0,
    #     simple_config.mcmc.burn_in,
    #     burnin_body,
    #     init_carry
    # )

###########################################################
    with TraceAnnotation("burn_in_phase"):
        for step in range(simple_config.mcmc.burn_in):
            print("Step burn in ", step)
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove = pmap_mcmc_step(params, walker_state, subkey)
            # Ensure all device work is finished before continuing
            walker_state = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, walker_state)
            local_mean_energy = dmc_sample.weighted_mean_energy(walker_state=walker_state)
            energy_history.at[step % len(energy_history)].set(local_mean_energy)
            walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=5000, use_external_energy=True, external_energy=local_mean_energy)
        walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=1, reweight_interval=1, use_external_energy=True, external_energy=local_mean_energy)            
        # Final sync after burn-in
        walker_state = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, walker_state)
        end = time.time()
        elapsed = end - start
        print(f"Elapsed time: {elapsed:.6f} seconds")

    with TraceAnnotation("main_dmc_loop"):
        with log_manager.create_writer() as writer:
            renormal_interval = 100
            energy_update_interval = 1000
            with TraceAnnotation("main_dmc_iteration"):
                for step in range(simple_config.mcmc.iteration):
                    print("Step ", step)
                    with TraceAnnotation(f"dmc_step_{step}"):
                        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
                        walker_state, pmove  = pmap_mcmc_step(params, walker_state, subkey)
                        # Ensure all device work is finished before continuing
                        walker_state = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, walker_state)
                        with TraceAnnotation("energy_calculation"):
                            local_mean_energy = dmc_sample.weighted_mean_energy(walker_state=walker_state)       
                        energy_history.at[step % len(energy_history)].set(local_mean_energy)
                        hist_mean_energy = jnp.mean(energy_history)
                        with TraceAnnotation("mean_energy_update"):
                            walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=energy_update_interval,reweight_interval=renormal_interval,use_external_energy=True, external_energy=local_mean_energy)
                        with TraceAnnotation("logging"):
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
if __name__=="__main__":
    Ne = 4
    dmc_iteration = 6000
    config = Config(network=Network(
            # type=NetworkType.laughlin
            type=NetworkType.parton,
            parton=PartonNetwork(
                fermionic_type=FermionicType.pfaffian,
                flux_type=FluxType.original_jastrow
            )
        ))
    config.seed = 126
    config.system.nspins = (Ne, 0)
    config.system.flux = 2 * Ne + 1
    config.system.tau = 0.0005
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    # config.optim.iterations = 20000
    config.batch_size = 32
    config.mcmc.width = 0.3
    config.initial_energy = config.system.nspins[0] * 0.5 + 0.467 * config.system.nspins[0] * config.system.interaction_strength

    config.log.pretrained_path = "../logs/pfaf_4_kappa_1.0/ckpt_000499.npz"
    config.log.save_path = "../logs/pfaf_4_kappa_1.0_dmc"
    # config.log.pretrained_path = "../logs/psiformer_4_kappa_1.0/ckpt_000519.npz"
    # config.log.save_path = "../logs/psiformer_4_kappa_1.0_dmc"
    # config.log.pretrained_path = "../logs/laughlin_4_kappa_1.0/ckpt_003884.npz"
    # config.log.save_path = f"../logs/laughlin_4_kappa_{config.system.interaction_strength}_dmc"
    # config.log.pretrained_path = f"../logs/psiformer25_{Ne}_kappa_1.0_dmc/ckpt_00{dmc_iteration-1}.npz"
    # config.log.save_path = f"../logs/psiformer25_{Ne}_kappa_1.0_dmc/dmc_run/"
    
    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 10
    config.mcmc.iteration = 20

    # Create profiling directory
    import os
    profile_dir = f"../logs/dmc_profiling_{int(time.time())}"
    os.makedirs(profile_dir, exist_ok=True)
    
    # Run with JAX profiler trace
    with trace(profile_dir, create_perfetto_link=False, create_perfetto_trace=False):
        run_dmc(config)