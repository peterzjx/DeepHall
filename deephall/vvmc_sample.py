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
from deephall.velocity_networks import make_v_network
from deephall.vvmc import vvmc
from chex import ArrayTree
import deephall.vvmc.velocity_utils as v_utils
from pathlib import Path
from upath import UPath
from jax import lax
logger = logging.getLogger("deephall")


def init_guess(key: PRNGKey, batch: int, nelec: int):
    """Create uniform samples on the sphere.

    Args:
        key: random key.
        batch: number of samples to generate.
        nelec: number of electrons.

    Returns:
        Electron coordinates of shape [batch, nelec, 2]
    """
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def initalize_state(cfg: Config, model: nn.Module):
    key_data, key_params = jax.random.split(jax.random.PRNGKey(cfg.seed))
    coords = init_guess(key_data, cfg.batch_size, sum(cfg.system.nspins))
    coords = coords.reshape((jax.device_count(), -1, *coords.shape[-2:]))
    ##############################################################################
    # theta = jnp.array([1.6856816, 2.4018655, 1.5067337, 0.42490557])
    # phi = jnp.array([-2.8268971, -2.2615817, -0.6118226, -2.640316])
    # electrons = jnp.stack([theta, phi], axis=-1)  # shape: (Ne, 2)
    # coords = jnp.stack([electrons] * jax.device_count(), axis=0)[:, jnp.newaxis, :, :]
    # print('xxx', coords.shape)
    ##############################################################################
    v_0 = jnp.ones_like(coords, dtype=jnp.complex64)
    logpsi_0 = jnp.zeros(coords.shape[:-2])
    print('init shape', coords.shape, v_0.shape, logpsi_0.shape)
    print('device #', jax.devices(), jax.device_count())

    d_0 = v_utils.calculate_d_metric_xy(coords, _2Q=cfg.system.flux)
    
    # Create walker state before replication
    
    # Initialize and replicate parameters
    params = model.init(key_params, coords[0, 0])

    dmc_state = DMCCheckpointState(
        params=kfac_jax.utils.replicate_all_local_devices(params),
        electrons=coords,
        electrons_xy=coords,
        d_metric=d_0,
        v=v_0,
        lnpsi=logpsi_0,
        local_energy=jnp.zeros_like(logpsi_0),  # TODO: calculate local energy
        weights=jnp.ones_like(logpsi_0),
        dmc_mean_energy=jnp.ones_like(logpsi_0)*cfg.initial_energy,
        dmc_run_step=jnp.zeros_like(logpsi_0),
        opt_state=None
    )

    return 0, dmc_state

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
    v_0 = jnp.ones_like(coords, dtype=jnp.complex64)
    logpsi_0 = jnp.zeros(coords.shape[:-2])
    d_0 = v_utils.calculate_d_metric(coords)
    
    with ckpt_path.open("rb") as npf, np.load(npf, allow_pickle=True) as f:
        step = f["step"].tolist() + 1
        params = f["params"].tolist()
        # logger.info("Restored checkpoint %s", ckpt_path)
        dmc_state = DMCCheckpointState(
        params=kfac_jax.utils.replicate_all_local_devices(params),
        electrons=coords,
        electrons_xy=coords,
        d_metric=d_0,
        v=v_0,
        lnpsi=logpsi_0,
        local_energy=jnp.zeros_like(logpsi_0),  # TODO: calculate local energy
        weights=jnp.ones_like(logpsi_0),
        dmc_mean_energy=jnp.zeros_like(logpsi_0),
        dmc_run_step=jnp.zeros_like(logpsi_0),
        opt_state=None
    )
        return step, dmc_state

def setup_mcmc(cfg: Config, network: LogPsiNetwork):
    # NOTE: we will takek batch_grad_fn inside, so we only need to pass the non-batched network
    mcmc_step = vvmc.make_vvmc_step(
        cfg.system,
        network,
        batch_per_device=cfg.batch_size // jax.device_count(),
        steps=cfg.mcmc.steps
    )
    
    pmap_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    return pmap_mcmc_step, pmoves

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
        v=walker_state.v,
        lnpsi=walker_state.lnpsi,
        local_energy=walker_state.local_energy,
        dmc_mean_energy = jnp.ones_like(walker_state.dmc_mean_energy ) * dmc_mean_energy_new,
        weights=walker_state.weights,
        d_metric=walker_state.d_metric,
        dmc_run_step=walker_state.dmc_run_step
    )
    return walker_state
