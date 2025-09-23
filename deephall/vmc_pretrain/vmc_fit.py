# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import signal
import sys
import time
from argparse import ArgumentParser
from typing import cast

import jax
import kfac_jax
import numpy as np
from chex import PRNGKey
from flax import linen as nn
from jax import numpy as jnp
from omegaconf import OmegaConf

from deephall import constants, mcmc, optimizers
from deephall.config import Config, OptimizerName
from deephall.log import LogManager, init_logging
from deephall.types import CheckpointState, DMCCheckpointState, WalkerState, get_walker_state, update_from_walker_state
from deephall.loss import LossMode, make_loss_fn
from deephall.networks import make_network
from deephall.types import LogPsiNetwork
from deephall.train import init_guess, setup_mcmc
from deephall.vmc_sample import initalize_state, restore_checkpoint
from pathlib import Path
from upath import UPath
logger = logging.getLogger("deephall")


def vmc_fit(laughlin_cfg: Config, cfg: Config):
    init_logging()
    log_manager = LogManager(cfg)
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(cfg.mcmc.width))
    ## Model loading
    laughlin_model = make_network(laughlin_cfg.system, laughlin_cfg.network)
    laughlin_network = cast(LogPsiNetwork, laughlin_model.apply)
    
    model = make_network(cfg.system, cfg.network)
    network = cast(LogPsiNetwork, model.apply)
        
    pmap_mcmc_step, pmove = setup_mcmc(laughlin_cfg, laughlin_network)
    print('initial setup_mcmc done', pmap_mcmc_step)
    if cfg.log.pretrained_path is not None:
        print('Restoring from pretrained path:', cfg.log.pretrained_path)
        initial_step, state = (
            initalize_state(cfg, model)
        )
        _, state = (
            restore_checkpoint(cfg, cfg.log.pretrained_path)
        )
    else:
        print('Training from scratch')
        initial_step, state = (
            initalize_state(cfg, model)
        )
    print('initial initalize_state done', state._fields)
    walker_state = get_walker_state(state) #WalkerState
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    opt_init, vmc_fit_training_step = optimizers.make_optimizer_vmc_fit_step(cfg, network)

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons, walker_state.lnpsi)))

    logger.info("Start VVMC with %s JAX devices", jax.device_count())


    if initial_step == 0:
        print("Burn-in ...")
        data = walker_state.electrons
        print('data.shape', data.shape)
        for step in range(cfg.mcmc.burn_in):
            data = walker_state.electrons
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            new_data, new_log_wfn, pmove = pmap_mcmc_step(state.params, data, subkey, mcmc_width)
            walker_state = walker_state._replace(electrons=new_data, lnpsi=new_log_wfn)
            
        logger.info("Burn in DMC complete")
        print("Done")
        
    state = update_from_walker_state(state, walker_state)
    data = state.electrons
    # walker_state = walker_state._replace(electrons=data, lnpsi=new_log_wfn)
    walker_state = get_walker_state(state)

    last_save_time = time.time()
    killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        for step in range(initial_step, cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            data, new_log_wfn, pmove = pmap_mcmc_step(state.params, data, subkey, mcmc_width)
            walker_state = walker_state._replace(electrons=data, lnpsi=new_log_wfn)
            state = update_from_walker_state(state, walker_state)
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            state, stats = vmc_fit_training_step(state, subkey)       
            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.4f}",
                loss=f"{stats['loss'][0]}",   
            )
            current_time = time.time()
            if (
                (
                    current_time - last_save_time > cfg.log.save_time_interval
                    and (step + 1) % cfg.log.save_step_interval == 0
                )
                or step == cfg.optim.iterations - 1
                or killer.kill_now
            ):
                writer.force_flush()
                log_manager.save_dmc_checkpoint(step, state)
                # log_manager.save_checkpoint(step, state)

class GracefulKiller:
    """Capture SIGINT and SIGTERM so that we can save checkpoints before exit."""

    kill_now = False

    def __init__(self):
        self.original_int = signal.signal(signal.SIGINT, self.exit_gracefully)
        self.original_term = signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Mark as exit and restore signal handlers."""
        del signum, frame
        if self.kill_now:  # Only handle the first signal
            return
        print("\r", end="")  # Clear ^C
        signal.signal(signal.SIGINT, self.original_int)
        signal.signal(signal.SIGTERM, self.original_term)
        self.kill_now = True
