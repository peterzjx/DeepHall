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
from deephall.types import LogPsiNetwork,WalkerState,CheckpointState, DMCCheckpointState
from deephall.loss import LossMode, make_loss_fn
from chex import ArrayTree
from pathlib import Path
from upath import UPath
from jax import lax
from deephall.train import initalize_state, init_guess
logger = logging.getLogger("deephall")


def restore_checkpoint(cfg: Config, ckpt: str | Path | UPath) -> tuple[int, DMCCheckpointState]:
    """Resore a given checkpoint.

    Args:
        ckpt: Checkpoint path.

    Returns:
        A tuple containing current step and state.
    """
    ckpt_path = UPath(ckpt)
    key_data, key_params = jax.random.split(jax.random.PRNGKey(cfg.seed))
    coords = init_guess(key_data, cfg.batch_size, sum(cfg.system.nspins))
    coords = coords.reshape((jax.device_count(), -1, *coords.shape[-2:]))
    
    with ckpt_path.open("rb") as npf, np.load(npf, allow_pickle=True) as f:
        step = f["step"].tolist() + 1
        params = f["params"].tolist()
        # logger.info("Restored checkpoint %s", ckpt_path)
        dmc_state = DMCCheckpointState(
            params=params,
            electrons=coords,
            electrons_xy=None,
            electrons_xy_move=None,
            d_metric=None,
            last_v=None,
            v=None,
            lnpsi=jnp.zeros(coords.shape[:-2]),
            local_energy=None,  # TODO: calculate local energy
            weights=None,
            dmc_mean_energy=None,
            dmc_run_step=None,
            opt_state=None
        )
        return step, dmc_state

@jax.jit
def weighted_mean_energy(walker_state: WalkerState):
    weighted_energy = jnp.sum(walker_state.weights * walker_state.local_energy) / jnp.sum(walker_state.weights)
    return weighted_energy


def update_mean_energy(walker_state: WalkerState, step: int, update_interval: int, reweight_interval: int=10, use_external_energy: bool=False, external_energy: float=0.0):
    weighted_energy = weighted_mean_energy(walker_state)
    dmc_mean_energy_new = 0.5 * (jnp.mean(walker_state.dmc_mean_energy) + weighted_energy)
    
    walker_state = WalkerState(
        electrons=walker_state.electrons,
        electrons_xy=walker_state.electrons_xy,
        electrons_xy_move=walker_state.electrons_xy_move,
        v=walker_state.v,
        lnpsi=walker_state.lnpsi,
        local_energy=walker_state.local_energy,
        dmc_mean_energy = jnp.ones_like(walker_state.dmc_mean_energy ) * dmc_mean_energy_new,
        weights=walker_state.weights,
        d_metric=walker_state.d_metric,
        dmc_run_step=walker_state.dmc_run_step
    )
    return walker_state
