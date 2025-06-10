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
from deephall.networks import make_network
from deephall.types import LogPsiNetwork, CheckpointState, DMCCheckpointState, WalkerState, get_walker_state, update_from_walker_state
from deephall.dmc import dmc
from deephall import dmc_sample
from deephall.train import initalize_state

logger = logging.getLogger("deephall")


def dmc_train(cfg: Config):
    init_logging()
    log_manager = LogManager(cfg)
    ## Model loading
    model = make_network(cfg.system, cfg.network)
    network = cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmove = dmc_sample.setup_mcmc(cfg, network)
    print('initial setup_mcmc done', pmap_mcmc_step)
    # assert cfg.log.pretrained_path is not None
    initial_step, state = (
        dmc_sample.initalize_state(cfg, model)
    )
    walker_state = get_walker_state(state)
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = jnp.stack([state.weights, state.local_energy], axis= -1)

    opt_init, dmc_training_step = optimizers.make_optimizer_step(cfg, network)

    # if cfg.log.pretrained_path is not None:
    #     initial_step, (params, data, opt_state, mcmc_width) = (
    #         initalize_state(cfg, model)
    #     )
    #     _, (params, _, opt_state, _) = (
    #         log_manager.try_load_pretrained_checkpoint()
    #     )
    # else:
    #     initial_step, (params, data, opt_state, mcmc_width) = (
    #         log_manager.try_restore_checkpoint() or initalize_state(cfg, model)
    #     )

    if (
        cfg.optim.optimizer == OptimizerName.none
        and cfg.log.restore_path is not None
        and cfg.log.restore_path != cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, walker_state.electrons))

    logger.info("Start DMC with %s JAX devices", jax.device_count())


    if initial_step == 0:
        for step in range(cfg.mcmc.burn_in):
            # sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            # data, pmove = pmap_mcmc_step(params, data, subkey, mcmc_width)

            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            # data, pmove= pmap_mcmc_step(params, data, subkey)
            walker_state, pmove, _, _, _, _, _, _, _  = pmap_mcmc_step(state.params, walker_state, subkey)
            # energy_history, mean_energy = dmc_sample.accumulate_energy(walker_state, energy_history, 1000)
            # walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=100, use_external_energy=True, external_energy=mean_energy)
        logger.info("Burn in DMC complete")
        # if cfg.log.initial_energy:
        #     # Logging inital energy is helpful for debugging. If we have initial energy
        #     # but have error in training, it's probably optimizer's fault
        #     initial_stats, _ = constants.pmap(
        #         make_loss_fn(network, cfg.system, LossMode.ENERGY_DIFF)
        #     )(params, data)
        #     logger.info("Initial energy: %s", initial_stats["energy"][0].real)

    state = update_from_walker_state(state, walker_state)

    last_save_time = time.time()
    killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        writer.hide("kinetic", "potential", "Lz_square")
        for step in range(initial_step, cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, _, _, _, _, _, _, _  = pmap_mcmc_step(state.params, walker_state, subkey)
            # new_data, pmove  = pmap_mcmc_step(state.params, state.data, subkey, mcmc_width)
            state = update_from_walker_state(state, walker_state)

            # energy_history, mean_energy = dmc_sample.accumulate_energy(walker_state, energy_history, 1000)
            # walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=100, use_external_energy=True, external_energy=mean_energy)

            # dmc_state = dmc_state._replace(walker_state=walker_state)
            print('training step ++')
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            state, stats = dmc_training_step(state, subkey)
            print('after dmc training step ++')
            writer.log(
                # step=str(step),
                # pmove=f"{pmove[0]:.2f}",
                # energy=f"{stats['energy'].real[0]:.4f}",
                # energy_imag=f"{stats['energy'].imag[0]:+.4f}",
                # potential=f"{stats['potential'][0]:.4f}",
                # kinetic=f"{stats['kinetic'].real[0]:.4f}",
                # variance=f"{stats['variance'][0]:.4f}",
                # Lz=f"{stats['angular_momentum_z'][0]:+.4f}",
                # Lz_square=f"{stats['angular_momentum_z_square'][0]:.4f}",
                # L_square=f"{stats['angular_momentum_square'][0]:.4f}",
            )
            current_time = time.time()
            if (
                (
                    current_time - last_save_time > cfg.log.save_time_interval
                    and (step + 1) % cfg.log.save_step_interval == 0
                )
                or jnp.isnan(stats["energy"].real).any()
                or step == cfg.optim.iterations - 1
                or killer.kill_now
            ):
                last_save_time = current_time
                writer.force_flush()
                # log_manager.save_checkpoint(step, state)
            if killer.kill_now or jnp.isnan(stats["energy"].real).any():
                raise SystemExit("=" * 30 + " ABORT " + "=" * 30)

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
