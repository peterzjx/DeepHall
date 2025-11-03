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
import time
from typing import cast

import jax
import kfac_jax
from jax import numpy as jnp
import numpy as np

import deephall.vmc_pretrain.training_step as training_step
from deephall.config import Config
from deephall.log import LogManager, init_logging, dedup_pytree
from deephall.types import get_walker_state, update_from_walker_state
from deephall.networks import make_network
from deephall.types import LogPsiNetwork
from deephall.train import setup_mcmc
from deephall.vmc_sample import initalize_state, restore_checkpoint
from deephall.graceful_killer import GracefulKiller


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
    logging.info('initial setup_mcmc done', pmap_mcmc_step)
    if cfg.log.pretrained_path is not None:
        logging.info('Restoring from pretrained path:', cfg.log.pretrained_path)
        initial_step, state = (
            initalize_state(cfg, model)
        )
        _, state = (
            restore_checkpoint(cfg, cfg.log.pretrained_path)
        )
    else:
        logging.info('Training from scratch')
        initial_step, state = (
            initalize_state(cfg, model)
        )
    logging.info('initial initalize_state done', state._fields)
    walker_state = get_walker_state(state)
    key = jax.random.PRNGKey(cfg.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    opt_init, vmc_fit_training_step = training_step.make_training_step_vmc_fit(cfg, network)

    # update opt state
    if state.opt_state is None:
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state = state._replace(opt_state=opt_init(state.params, subkey, (walker_state.electrons, walker_state.lnpsi)))

    logging.info("Start VVMC with %s JAX devices", jax.device_count())


    if initial_step == 0:
        logging.info("Burn-in ...")
        data = walker_state.electrons
        for step in range(cfg.mcmc.burn_in):
            data = walker_state.electrons
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            new_data, new_log_wfn, pmove = pmap_mcmc_step(state.params, data, subkey, mcmc_width)
            walker_state = walker_state._replace(electrons=new_data, lnpsi=new_log_wfn)
        logging.info("Burn in DMC complete")
        
    state = update_from_walker_state(state, walker_state)
    data = state.electrons
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

            # Save coordinates independently if configured to do so. This allows
            # saving coords every N iterations even when checkpoints are less
            # frequent.
            try:
                save_coords_enabled = getattr(cfg.log, "save_coords", False)
                if save_coords_enabled:
                    save_coords_interval = getattr(cfg.log, "save_coords_step_interval", None)
                    if save_coords_interval is None:
                        save_coords_interval = cfg.log.save_step_interval
                    if save_coords_interval is not None and save_coords_interval > 0:
                        if ((step + 1) % save_coords_interval == 0) or (
                            step == cfg.optim.iterations - 1
                        ) or killer.kill_now:
                            coords = dedup_pytree(state.electrons)
                            coords = jax.device_get(coords)
                            loss_val = float(jax.device_get(stats["loss"])[0])
                            coords_path = log_manager.save_path / f"coords_{step:06d}.npz"
                            with coords_path.open("wb") as f:
                                np.savez_compressed(f, loss=loss_val, electrons=np.asarray(coords))
                            logging.info("Saved coordinates and loss to %s", coords_path)
            except Exception as e:
                logging.warning("Failed saving coordinates: %s", e)

            # Checkpoint save uses both a time interval and a step interval, or
            # triggers at the final step / graceful kill.
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
                # update last_save_time so time-based interval is enforced
                last_save_time = current_time

