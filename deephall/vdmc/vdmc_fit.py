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
from deephall import vdmc_sample
from deephall.train import initalize_state

logger = logging.getLogger("deephall")

def get_laughlin_cfg(cfg: Config):
    config = Config()
    config.network.type = cfg.network.type
    config.seed = 1
    config.system.nspins = cfg.system.nspins
    config.system.flux = cfg.system.flux
    config.system.interaction_strength = cfg.system.interaction_strength
    config.optim.iterations = cfg.optim.iterations
    config.batch_size = cfg.batch_size
    config.mcmc.burn_in = cfg.mcmc.burn_in
    config.mcmc.iteration = cfg.mcmc.iteration
    config.initial_energy = cfg.initial_energy
    config.log.initial_energy = False
    config.log.save_path = cfg.log.save_path
    return config

def vdmc_fit(cfg: Config):
    init_logging()
    log_manager = LogManager(cfg)
    laughlin_cfg = get_laughlin_cfg(cfg)
    ## Model loading
    laughlin_model = make_v_network(laughlin_cfg.system, laughlin_cfg.network)
    model = make_v_network(cfg.system, cfg.network)

    network = cast(LogPsiNetwork, model.apply)
    laughlin_model = cast(LogPsiNetwork, laughlin_model.apply)
    
    pmap_mcmc_step, pmove = vdmc_sample.setup_mcmc(laughlin_cfg, laughlin_model)
    print('initial setup_mcmc done', pmap_mcmc_step)
    # assert cfg.log.pretrained_path is not None
    if cfg.log.pretrained_path is not None:
        initial_step, state = (
            vdmc_sample.initalize_state(cfg, model)
        )
        _, state = (
            vdmc_sample.restore_checkpoint(cfg, cfg.log.pretrained_path)
        )
    else:
        initial_step, state = (
            vdmc_sample.initalize_state(cfg, model)
        )
    walker_state = get_walker_state(state)
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    energy_history = None

    opt_init, vdmc_fit_step = optimizers.make_optimizer_dmc_step(cfg, network)

    if (
        cfg.optim.optimizer == OptimizerName.none
        and cfg.log.restore_path is not None
        and cfg.log.restore_path != cfg.log.save_path
    ):  # Reset steps because inference run is another run
        initial_step = 0

    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons, walker_state.weights)))

    logger.info("Start DMC with %s JAX devices", jax.device_count())

    if initial_step == 0:
        for step in range(cfg.mcmc.burn_in):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            # energy_history, mean_energy = vdmc_sample.accumulate_energy(walker_state, energy_history, 1000)
            # walker_state, changed, _, _, _ = vdmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=5000, use_external_energy=True, external_energy=mean_energy)
        # walker_state, _, _, _, _ = vdmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=1, reweight_interval=1, use_external_energy=True, external_energy=mean_energy)            
        energy_history = None
        logger.info("Burn in DMC complete")
        
    state = update_from_walker_state(state, walker_state)

    killer = GracefulKiller()
    with log_manager.create_writer() as writer:
        writer.hide("kinetic", "potential", "Lz_square")
        renormal_interval = 100
        energy_update_interval = 1000
        for step in range(initial_step, cfg.optim.iterations):
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold = pmap_mcmc_step(state.params, walker_state, subkey)
            # energy_history, mean_energy = vdmc_sample.accumulate_energy(walker_state, energy_history, max_length=10000)
            # walker_state, changed, idx_min, conditioned, change_shape = vdmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=energy_update_interval,reweight_interval=renormal_interval,use_external_energy=True, external_energy=mean_energy)
            state = update_from_walker_state(state, walker_state)
            if step%renormal_interval==0 and (jnp.min(walker_state.weights)<0.01 or jnp.max(walker_state.weights)>5.0) and renormal_interval>10:
                renormal_interval = renormal_interval - 1
            assert renormal_interval>10
            
            
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            state, stats = vdmc_fit_step(state, subkey)
            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                local_energy=f"{vdmc_sample.weighted_mean_energy(walker_state):.6f}",
                dmc_mean_energy=f"{jnp.mean(walker_state.dmc_mean_energy):.6f}",
                history_mean_energy=f"{mean_energy:.6f}",
                weight_max=f"{jnp.max(walker_state.weights):.6f}",
                weight_min=f"{jnp.min(walker_state.weights):.6f}",
                weight_std=f"{jnp.std(walker_state.weights):.6f}",
            )
            
            # writer.log(
            #     # step=str(step),
            #     # pmove=f"{pmove[0]:.2f}",
            #     # energy=f"{stats['energy'].real[0]:.4f}",
            #     # # energy_imag=f"{stats['energy'].imag[0]:+.4f}",
            #     # # potential=f"{stats['potential'][0]:.4f}",
            #     # # kinetic=f"{stats['kinetic'].real[0]:.4f}",
            #     # # variance=f"{stats['variance'][0]:.4f}",
            #     # # Lz=f"{stats['angular_momentum_z'][0]:+.4f}",
            #     # # Lz_square=f"{stats['angular_momentum_z_square'][0]:.4f}",
            #     # # L_square=f"{stats['angular_momentum_square'][0]:.4f}",
            #     step=str(step),
            #     pmove=f"{pmove[0]:.2f}",
            #     local_energy=f"{vdmc_sample.weighted_mean_energy(walker_state):.6f}",
            #     dmc_mean_energy=f"{jnp.mean(walker_state.dmc_mean_energy):.6f}",
            #     history_mean_energy=f"{mean_energy:.6f}"
            # )
            current_time = time.time()
            if (
                (
                    (step + 1) % cfg.log.save_step_interval == 0
                )
                or jnp.isnan(stats["energy"].real).any()
                or step == cfg.optim.iterations - 1
                or killer.kill_now
            ):
                last_save_time = current_time
                writer.force_flush()
                log_manager.save_dmc_checkpoint(step, state)
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
