import jax
import jax.numpy as jnp

import numpy as np
from deephall import dmc_sample, Config
from deephall import constants, mcmc
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
from deephall.optimizers.kfac import GRAPH_PATTERNS
import deephall.vvmc.training_step as training_step
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
    config.system.interaction_strength = 8.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    config.optim.optimizer = OptimizerName.adam
    config.optim.iterations = 10000
    config.batch_size = 48
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    config.log.pretrained_path = "../logs/super_laughlin_v_fit/ckpt_004999.npz"
    config.log.save_path = "../logs/super_laughlin_v_train_k8"
    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 10
    return config

def test_vvmc_train(simple_cfg: Config):
    init_logging()
    log_manager = LogManager(simple_cfg)
    ## Model loading
    model = make_v_network(simple_cfg.system, simple_cfg.network)

    network = cast(LogPsiNetwork, model.apply)
    # laughlin_model = cast(LogPsiNetwork, laughlin_model.apply)
    
    pmap_mcmc_step, pmove = vvmc_sample.setup_mcmc(simple_cfg, network)
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
    key = jax.random.PRNGKey(simple_cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = None

    opt_init, vvmc_training_step = training_step.make_training_step_vvmc(simple_cfg, network, GRAPH_PATTERNS)

    if (
        simple_cfg.optim.optimizer == OptimizerName.none
        and simple_cfg.log.restore_path is not None
        and simple_cfg.log.restore_path != simple_cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons_xy, walker_state.electrons_xy_move)))
    

    logger.info("Start VVMC Energy training with %s JAX devices", jax.device_count())

    if initial_step == 0:
        for step in range(simple_cfg.mcmc.burn_in):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            print("Burn-in step # ", step)
        energy_history = None
        logger.info("Burn in DMC complete")
        
    state = update_from_walker_state(state, walker_state)

    # # killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        # writer.hide("kinetic", "potential", "Lz_square")
        for step in range(initial_step, simple_cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            state = update_from_walker_state(state, walker_state)
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            state, stats = vvmc_training_step(state, subkey)        

            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                # electrons_xy=f"{walker_state.electrons_xy[0]}",
                # v=f"{walker_state.v[0]}",
                # target=f"{stats['target'][0]}",
                # prediction=f"{stats['prediction'][0]}",
                energy=f"{stats['energy'][0].real}",   
                kinetic=f"{stats['kinetic'][0].real}",   
                potential=f"{stats['potential'][0].real}",   
                # gradient=f"{stats['gradient']}",             
            )
            
            # current_time = time.time()
            # if (
            #     (
            #         (step + 1) % cfg.log.save_step_interval == 0
            #     )
            #     or jnp.isnan(stats["energy"].real).any()
            #     or step == cfg.optim.iterations - 1
            #     or killer.kill_now
            # ):
            #     last_save_time = current_time
            #     writer.force_flush()
            #     log_manager.save_dmc_checkpoint(step, state)
            # if killer.kill_now or jnp.isnan(stats["energy"].real).any():
            #     raise SystemExit("=" * 30 + " ABORT " + "=" * 30)