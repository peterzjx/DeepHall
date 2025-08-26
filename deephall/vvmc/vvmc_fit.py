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
from deephall.config import Config, OptimizerName, NetworkType
from deephall.log import LogManager, init_logging
from deephall.velocity_networks import make_v_network
from deephall.types import LogPsiNetwork, CheckpointState, DMCCheckpointState, WalkerState, get_walker_state, update_from_walker_state
from deephall import vvmc_sample
from deephall.train import initalize_state

logger = logging.getLogger("deephall")

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

def vvmc_fit(cfg: Config):
    init_logging()
    log_manager = LogManager(cfg)
    laughlin_cfg = get_laughlin_cfg(cfg)
    ## Model loading
    laughlin_model = make_v_network(laughlin_cfg.system, laughlin_cfg.network)
    model = make_v_network(cfg.system, cfg.network)

    network = cast(LogPsiNetwork, model.apply)
    laughlin_network = cast(LogPsiNetwork, laughlin_model.apply)
    
    pmap_mcmc_step, pmove = vvmc_sample.setup_mcmc(laughlin_cfg, laughlin_network)
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
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = None

    opt_init, vvmc_fit_training_step = optimizers.make_optimizer_vvmc_fit_step(cfg, network)

    if (
        cfg.optim.optimizer == OptimizerName.none
        and cfg.log.restore_path is not None
        and cfg.log.restore_path != cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons_xy, walker_state.v)))

    logger.info("Start VVMC with %s JAX devices", jax.device_count())

    if initial_step == 0:
        print("Burn-in ...")
        for step in range(cfg.mcmc.burn_in):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, _, _ = pmap_mcmc_step(state.params, walker_state, subkey)
            
        logger.info("Burn in DMC complete")
        print("Done")
        
    state = update_from_walker_state(state, walker_state)

    # # killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        writer.hide("kinetic", "potential", "Lz_square")
        for step in range(initial_step, cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            state = update_from_walker_state(state, walker_state)
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            # print("Fitting total mini-step # ", step)
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
                    (step + 1) % cfg.log.save_step_interval == 0
                )
                or step == cfg.optim.iterations - 1
            ):
                writer.force_flush()
                log_manager.save_dmc_checkpoint(step, state)

def vvmc_reverse_fit(cfg: Config):
    init_logging()
    log_manager = LogManager(cfg)
    laughlin_cfg = get_laughlin_cfg(cfg)
    ## Model loading
    laughlin_model = make_v_network(laughlin_cfg.system, laughlin_cfg.network)
    model = make_v_network(cfg.system, cfg.network)

    network = cast(LogPsiNetwork, model.apply)
    laughlin_network = cast(LogPsiNetwork, laughlin_model.apply)
    
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
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = None

    opt_init, vvmc_fit_training_step = optimizers.make_optimizer_vvmc_fit_step(cfg, laughlin_network)

    if (
        cfg.optim.optimizer == OptimizerName.none
        and cfg.log.restore_path is not None
        and cfg.log.restore_path != cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons_xy, walker_state.v)))

    logger.info("Start VVMC with %s JAX devices", jax.device_count())

    if initial_step == 0:
        print("Burn-in ...")
        for step in range(cfg.mcmc.burn_in):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, _, _ = pmap_mcmc_step(state.params, walker_state, subkey)
            
        logger.info("Burn in DMC complete")
        print("Done")
        
    state = update_from_walker_state(state, walker_state)

    # # killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        writer.hide("kinetic", "potential", "Lz_square")
        for step in range(initial_step, cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            state = update_from_walker_state(state, walker_state)
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            state, stats = vvmc_fit_training_step(state, subkey)        

            # writer.log(
            #     step=str(step),
            #     pmove=f"{pmove[0]:.2f}",
            #     # electrons_xy=f"{walker_state.electrons_xy[0]}",
            #     # v=f"{walker_state.v[0]}",
            #     # target=f"{stats['target'][0]}",
            #     # prediction=f"{stats['prediction'][0]}",
            #     loss=f"{stats['loss'][0]}",   
            #     # gradient=f"{stats['gradient']}",             
            # )
            
            # current_time = time.time()
            # if (
            #     (
            #         (step + 1) % cfg.log.save_step_interval == 0
            #     )
            #     or step == cfg.optim.iterations - 1
            # ):
            #     writer.force_flush()
            #     log_manager.save_dmc_checkpoint(step, state)