import jax
import jax.numpy as jnp

import numpy as np
from deephall import dmc_sample, Config
from deephall import constants, mcmc, optimizers
from deephall.vvmc import vvmc_fit
from deephall.config import Network, NetworkType, System, PsiformerNetwork,Network, NetworkType, FluxType, FermionicType, PartonNetwork, OptimizerName
from deephall.types import CheckpointState, DMCCheckpointState, WalkerState, get_walker_state, update_from_walker_state
from pathlib import Path
from deephall.types import LogPsiNetwork
from deephall.loss import LossMode, make_loss_fn
from deephall.velocity_networks import make_v_network
import jax
import jax.numpy as jnp
import kfac_jax
from omegaconf import OmegaConf
import logging
import jax.numpy as jnp
from deephall.train import train
from typing import cast
import pytest
from pytest import CaptureFixture

import logging
from deephall.log import LogManager, init_logging
from deephall import vvmc_sample
import time
logger = logging.getLogger("deephall")
@pytest.fixture
def simple_cfg():
    config = Config(network=Network(
            type=NetworkType.super_laughlin_v
        ))
    config.seed = 564
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.0001
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    config.optim.iterations = 100
    config.optim.optimizer = OptimizerName.adam
    config.batch_size = 48
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    config.log.save_path = "../logs/super_laughlin_v_fit"
    config.mcmc.burn_in = 100
    return config

def get_laughlin_cfg(cfg: Config):
    config = Config()
    config.network.type = NetworkType.laughlin_v
    config.seed = 1
    config.system.nspins = cfg.system.nspins
    config.system.flux = cfg.system.flux
    config.system.interaction_strength = cfg.system.interaction_strength
    config.optim.iterations = cfg.optim.iterations
    config.batch_size = cfg.batch_size
    config.mcmc.burn_in = cfg.mcmc.burn_in
    config.initial_energy = cfg.initial_energy
    config.log.initial_energy = False
    config.log.save_path = cfg.log.save_path
    return config

def test_vvmc_fit(simple_cfg: Config):
    init_logging()
    log_manager = LogManager(simple_cfg)
    laughlin_cfg = get_laughlin_cfg(simple_cfg)
    ## Model loading
    laughlin_model = make_v_network(laughlin_cfg.system, laughlin_cfg.network)
    model = make_v_network(simple_cfg.system, simple_cfg.network)

    network = cast(LogPsiNetwork, model.apply)
    laughlin_model = cast(LogPsiNetwork, laughlin_model.apply)
    
    pmap_mcmc_step, pmove = vvmc_sample.setup_mcmc(laughlin_cfg, laughlin_model)
    print('initial setup_mcmc done', pmap_mcmc_step)
    if simple_cfg.log.pretrained_path is not None:
        initial_step, state = (
            vvmc_sample.initalize_state(simple_cfg, model)
        )
        _, state = (
            vvmc_sample.restore_checkpoint(simple_cfg, simple_cfg.log.pretrained_path)
        )
    else:
        initial_step, state = (
            vvmc_sample.initalize_state(simple_cfg, model)
        )
    walker_state = get_walker_state(state)
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(simple_cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = None

    opt_init, vvmc_fit_training_step = optimizers.make_optimizer_vvmc_fit_step(simple_cfg, network)

    if (
        simple_cfg.optim.optimizer == OptimizerName.none
        and simple_cfg.log.restore_path is not None
        and simple_cfg.log.restore_path != simple_cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons_xy, walker_state.v)))

    logger.info("Start VVMC with %s JAX devices", jax.device_count())

    if initial_step == 0:
        print("Burn-in ...")
        for step in range(simple_cfg.mcmc.burn_in):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, _, _ = pmap_mcmc_step(state.params, walker_state, subkey)
            
        logger.info("Burn in DMC complete")
        print("Done")
        
    state = update_from_walker_state(state, walker_state)

    # # killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        writer.hide("kinetic", "potential", "Lz_square")
        for step in range(initial_step, simple_cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            state = update_from_walker_state(state, walker_state)
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            print("Fitting total mini-step # ", step)
            # print('state:', state)
            # input()
            state, stats = vvmc_fit_training_step(state, subkey)        

            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                # electrons_xy=f"{walker_state.electrons_xy[0]}",
                # v=f"{walker_state.v[0]}",
                # target=f"{stats['target'][0]}",
                # prediction=f"{stats['prediction'][0]}",
                loss=f"{stats['loss'][0]}",   
                # gradient=f"{stats['gradient']}",             
            )
            
            current_time = time.time()
            if (
                (
                    (step + 1) % simple_cfg.log.save_step_interval == 0
                )
                # or jnp.isnan(stats["energy"].real).any()
                or step == simple_cfg.optim.iterations - 1
            ):
                # last_save_time = current_time
                writer.force_flush()
                log_manager.save_dmc_checkpoint(step, state)
            # if killer.kill_now or jnp.isnan(stats["energy"].real).any():
            #     raise SystemExit("=" * 30 + " ABORT " + "=" * 30)