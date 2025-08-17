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
from deephall.loss import LossMode, make_loss_fn
from deephall.velocity_networks import make_v_network
from deephall.types import LogPsiNetwork, CheckpointState, DMCCheckpointState, WalkerState, get_walker_state, update_from_walker_state
from deephall import vvmc_sample
from deephall.train import initalize_state

logger = logging.getLogger("deephall")


def vvmc_train(cfg: Config):
    init_logging()
    log_manager = LogManager(cfg)
    ## Model loading
    model = make_v_network(cfg.system, cfg.network)

    network = cast(LogPsiNetwork, model.apply)
    # laughlin_model = cast(LogPsiNetwork, laughlin_model.apply)
    
    pmap_mcmc_step, pmove = vvmc_sample.setup_mcmc(cfg, network)
    print('initial setup_mcmc done', pmap_mcmc_step)
    if cfg.log.pretrained_path is not None:
        initial_step, state = (
            vvmc_sample.initalize_state(cfg, model)
        )
        _, state = (
            vvmc_sample.restore_checkpoint(cfg, cfg.log.pretrained_path)
        )
    else:
        initial_step, state = (
            vvmc_sample.initalize_state(cfg, model)
        )
    walker_state = get_walker_state(state)
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = None

    opt_init, vvmc_training_step = optimizers.make_optimizer_vvmc_step(cfg, network)

    if (
        cfg.optim.optimizer == OptimizerName.none
        and cfg.log.restore_path is not None
        and cfg.log.restore_path != cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons_xy, walker_state.electrons_xy_move)))
    

    logger.info("Start VVMC Energy training with %s JAX devices", jax.device_count())

    if initial_step == 0:
        for step in range(cfg.mcmc.burn_in):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
        logger.info("Burn in VVMC complete")
        
    state = update_from_walker_state(state, walker_state)

    # # killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        # writer.hide("kinetic", "potential", "Lz_square")
        for step in range(initial_step, cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            state = update_from_walker_state(state, walker_state)
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            state, stats = vvmc_training_step(state, subkey)        

            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                energy=f"{stats['energy'][0].real}",   
                kinetic=f"{stats['kinetic'][0].real}",   
                potential=f"{stats['potential'][0].real}",        
            )