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
from deephall.types import CheckpointState, DMCCheckpointState
from deephall.loss import LossMode, make_loss_fn
from deephall.networks import make_network
from deephall.types import LogPsiNetwork
from deephall.dmc import dmc
from deephall.dmc.dmc import WalkerState
from chex import ArrayTree
import deephall.dmc.velocity_utils as v_utils
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
    v_0 = jnp.ones_like(coords)
    logpsi_0 = jnp.zeros(coords.shape[:-2])
    print('init shape', coords.shape, v_0.shape, logpsi_0.shape)
    print('device #', jax.devices(), jax.device_count())

    d_0 = v_utils.calculate_d_metric(coords)
    
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
    v_0 = jnp.ones_like(coords)
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
    if cfg.mcmc.use_dmc:
        # NOTE: we will takek batch_grad_fn inside, so we only need to pass the non-batched network
        mcmc_step = dmc.make_dmc_step(
            cfg.system,
            network,
            batch_per_device=cfg.batch_size // jax.device_count(),
            steps=cfg.mcmc.steps
        )
    else:
        batch_network = jax.vmap(network, in_axes=(None, 0))
        mcmc_step = mcmc.make_mcmc_step(
            batch_network,
            batch_per_device=cfg.batch_size // jax.device_count(),
            steps=cfg.mcmc.steps
        )
    # pmap_mcmc_step = constants.pmap(mcmc_step)
    pmap_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    return pmap_mcmc_step, pmoves


# def update_walker_state_from_pretrained(cfg: Config, model: nn.Module, params: ArrayTree, walker_state: WalkerState):
#     psi = v_utils.psi(params, model, walker_state.electrons)
#     v = v_utils.drift_velocity(params, model, walker_state.electrons)
#     energy = v_utils.local_energy(params, model, walker_state.electrons)
#     return WalkerState(
#         electrons=walker_state.electrons,
#         v=v,
#         psi=psi,
#         local_energy=energy,
#         weights=walker_state.weights
#     )

@jax.jit
def weighted_mean_energy(walker_state: WalkerState):
    weighted_energy = jnp.sum(walker_state.weights * walker_state.local_energy) / jnp.sum(walker_state.weights)
    return weighted_energy

def renormalize_weight(
    W: jnp.ndarray,
    energy: jnp.ndarray,
    coord: jnp.ndarray,
    coord_xy: jnp.ndarray,
    velocity: jnp.ndarray,
    lnpsi: jnp.ndarray,
    dmat: jnp.ndarray,
):
    """
    Applies normalization logic to W and replaces corresponding entries in
    energy, coord, coord_xy, velocity, lnpsi, and dmat based on value thresholds.
    W: [nwalkers]
    all other tensors: [nwalkers, ...]
    
    Find largest (idx_max) and smallest (idx_min) W values.
    If W[idx_max] is `large` and W[idx_min] is `small`, copy idx_max row to idx_min row
    and set both W[idx_max] and W[idx_min] to W[idx_max]/2.
    
    Returns:
        Tuple of updated arrays: (W, energy, coord, coord_xy, velocity, lnpsi, dmat, idx_max, idx_min, w_max, changed)
    """
    # Find indices of maximum and minimum W values
    idx_max = jnp.argmax(W)
    idx_min = jnp.argmin(W)
    
    # Get the actual values
    w_max = W[idx_max]
    w_min = W[idx_min]
    
    # Check condition: W[idx_max] is large and W[idx_min] is small
    condition = (w_max > 2.0) & (w_min < 0.1)
    # if not jnp.any(condition):
    #     return W_updated, energy_updated, coord_updated, coord_xy_updated, velocity_updated, lnpsi_updated, dmat_updated, idx_max, idx_min, w_max, changed, conditioned, change_shape
    conditioned = condition
    change_shape = W.shape
    
    # Calculate new weight value (will be used if condition is True)
    new_weight = w_max / 2.0
    
    # Create masks for the updates
    max_mask = jnp.arange(W.shape[0]) == idx_max
    min_mask = jnp.arange(W.shape[0]) == idx_min
    
    # Update W values conditionally
    W_updated = jnp.where(
        condition,
        jnp.where(max_mask | min_mask, new_weight, W),
        W
    )
    
    # Update other tensors conditionally
    energy_updated = jnp.where(
        condition,
        jnp.where(min_mask, energy[idx_max], energy),
        energy
    )
    
    coord_updated = jnp.where(
        condition,
        jnp.where(min_mask[:, None, None], coord[idx_max], coord),
        coord
    )
    
    coord_xy_updated = jnp.where(
        condition,
        jnp.where(min_mask[:, None, None], coord_xy[idx_max], coord_xy),
        coord_xy
    )
    
    velocity_updated = jnp.where(
        condition,
        jnp.where(min_mask[:, None, None], velocity[idx_max], velocity),
        velocity
    )
    
    lnpsi_updated = jnp.where(
        condition,
        jnp.where(min_mask, lnpsi[idx_max], lnpsi),
        lnpsi
    )
    
    dmat_updated = jnp.where(
        condition,
        jnp.where(min_mask[:, None, None], dmat[idx_max], dmat),
        dmat
    )
    
    # Return 1 if condition was met, 0 otherwise
    changed = jnp.where(condition, 1, 0)
    
    return W_updated, energy_updated, coord_updated, coord_xy_updated, velocity_updated, lnpsi_updated, dmat_updated, idx_max, idx_min, w_max, changed, conditioned, change_shape


def update_mean_energy(walker_state: WalkerState, step: int, update_interval: int, reweight_interval: int=10, use_external_energy: bool=False, external_energy: float=0.0):
    changed = 0
    idx_min = None
    conditioned = None
    change_shape = walker_state.weights.shape
    pmap_renormalize_weight = constants.pmap(renormalize_weight)
    dmc_mean_energy_new = walker_state.dmc_mean_energy
    renormalized = False
    if step % reweight_interval != 0 and step % update_interval != 0:
        return walker_state

    if step % update_interval == 0:
        if use_external_energy:
            weighted_energy = external_energy
        else:
            weighted_energy = weighted_mean_energy(walker_state)
        dmc_mean_energy_new = 0.5 * (jnp.mean(walker_state.dmc_mean_energy) + weighted_energy)

    if step % reweight_interval == 0:
        renormalized = True
        weights, local_energy, ele, ele_xy, velocity, lnpsi, dmat, idx_max, idx_min, w_max, changed, conditioned, change_shape = pmap_renormalize_weight(walker_state.weights, 
                                                                                        walker_state.local_energy,
                                                                                        walker_state.electrons,
                                                                                        walker_state.electrons_xy,
                                                                                        walker_state.v,
                                                                                        walker_state.lnpsi,
                                                                                        walker_state.d_metric)
        assert (weights.shape == walker_state.weights.shape)
        assert (local_energy.shape == walker_state.local_energy.shape), f'{local_energy.shape} != {walker_state.local_energy.shape}'
        assert (ele.shape == walker_state.electrons.shape)
        assert (ele_xy.shape == walker_state.electrons_xy.shape)
        assert (velocity.shape == walker_state.v.shape)
        assert (lnpsi.shape == walker_state.lnpsi.shape)
        assert (dmat.shape == walker_state.d_metric.shape)
    
    if renormalized == True:
        walker_state = WalkerState(
            electrons=ele,
            electrons_xy=ele_xy,
            v=velocity,
            lnpsi=lnpsi,
            local_energy=local_energy,
            dmc_mean_energy = jnp.ones_like(walker_state.dmc_mean_energy ) * dmc_mean_energy_new,
            weights=weights,
            d_metric=dmat,
            dmc_run_step=walker_state.dmc_run_step
        )
    else:
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

# def accumulate_energy(walker_state: WalkerState, energy_hist: jnp.ndarray, max_length: int):
    
#     new_hist = jnp.array([weighted_mean_energy(walker_state=walker_state)])
    
#     if energy_hist == None:
#         energy_hist = new_hist
#     else:
#         energy_hist = jnp.concatenate([energy_hist, new_hist], axis=0)
#         if energy_hist.shape[0]>max_length:
#             new_len = new_hist.shape[0]
#             energy_hist = energy_hist[new_len:]
    
#     mean_energy = jnp.mean(energy_hist)
#     return energy_hist, mean_energy
