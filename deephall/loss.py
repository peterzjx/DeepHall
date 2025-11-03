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

import enum
from collections.abc import Callable
from typing import cast

import jax
import kfac_jax
from chex import ArrayTree
from jax import numpy as jnp

from deephall import constants
from deephall.config import System
from deephall.hamiltonian import OtherObservables, local_energy, weighted_local_energy, local_v_energy
from deephall.types import LogPsiNetwork, LossStats
from jax import numpy as jnp

def iqr_clip_real(x: jnp.ndarray, scale=100.0) -> jnp.ndarray:
    q1 = jnp.nanquantile(x, 0.25)
    q3 = jnp.nanquantile(x, 0.75)
    iqr = q3 - q1
    return jnp.clip(x, q1 - scale * iqr, q3 + scale * iqr)


def iqr_clip(x: jnp.ndarray, scale=100.0) -> jnp.ndarray:
    return iqr_clip_real(x.real, scale) + 1j * iqr_clip_real(x.imag, scale)


class LossMode(enum.Enum):
    FUNCTION_OVLP = enum.auto()
    ENERGY_GRAD = enum.auto()
    ENERGY_DIFF = enum.auto()
    SR_F_VECTOR = enum.auto()


def make_loss_fn(
    network: LogPsiNetwork, system: System, mode: LossMode = LossMode.ENERGY_GRAD
) -> Callable[[ArrayTree, jnp.ndarray], tuple[LossStats, jnp.ndarray]]:
    loss_fn = local_energy(network, system)
    batch_local_energy = jax.vmap(loss_fn, in_axes=(None, 0))

    df_real = jax.vmap(
        jax.value_and_grad(lambda params, x: network(params, x).real), in_axes=(None, 0)
    )
    df_imag = jax.vmap(
        jax.value_and_grad(lambda params, x: network(params, x).imag), in_axes=(None, 0)
    )

    def loss_prod(grad_logpsi_conj, diff):
        diff = diff.reshape(
            diff.shape + (1,) * (len(grad_logpsi_conj.shape) - len(diff.shape))
        )
        return jnp.nan_to_num(2 * jnp.nanmean(grad_logpsi_conj * diff, axis=0))

    def loss_and_grad(params: ArrayTree, data: jnp.ndarray):
        el, other_observables = batch_local_energy(params, data)
        pmean_observables = cast(
            OtherObservables,
            jax.tree.map(lambda x: constants.pmean(jnp.mean(x)), other_observables),
        )

        loss = constants.pmean(jnp.nanmean(el))
        clipped_loss = constants.pmean(jnp.nanmean(iqr_clip(el)))
        diff_to_clip = el - clipped_loss
        if system.lz_penalty:
            lz_square = other_observables["angular_momentum_z_square"]
            lz = other_observables["angular_momentum_z"]
            clipped_lz_square = constants.pmean(jnp.nanmean(iqr_clip(lz_square)))
            clipped_lz = constants.pmean(jnp.nanmean(iqr_clip(lz)))
            diff_to_clip += system.lz_penalty * (
                (lz_square - clipped_lz_square)
                - 2 * system.lz_center * (lz - clipped_lz)
            )
        if system.l2_penalty:
            l2 = other_observables["angular_momentum_square"]
            clipped_l2 = constants.pmean(jnp.nanmean(iqr_clip(l2)))
            diff_to_clip += system.l2_penalty * (l2 - clipped_l2)
        diff = iqr_clip(diff_to_clip)

        variance = constants.pmean(jnp.nanmean(el.real**2) - loss.real**2)
        stats = LossStats(**pmean_observables, energy=loss, variance=variance)
        if mode == LossMode.ENERGY_DIFF:
            return stats, diff

        primal_real, tangent_real = df_real(params, data)
        _, tangent_imag = df_imag(params, data)
        kfac_jax.register_normal_predictive_distribution(primal_real[:, None])
        tangent_out = jax.tree.map(
            lambda real, imag: loss_prod(real - 1j * imag, diff),
            tangent_real,
            tangent_imag,
        )

        if mode == LossMode.ENERGY_GRAD:
            return stats, jax.tree.map(jnp.real, tangent_out)
        elif mode == LossMode.SR_F_VECTOR:
            return stats, tangent_out

    return loss_and_grad

def make_dmc_loss_fn(
    network: LogPsiNetwork, system: System, mode: LossMode = LossMode.ENERGY_GRAD
) -> Callable[[ArrayTree, jnp.ndarray], tuple[LossStats, jnp.ndarray]]:
    loss_fn = weighted_local_energy(network, system)
    batch_weighted_local_energy = jax.vmap(loss_fn, in_axes=(None, 0))

    df_real = jax.vmap(
        jax.value_and_grad(lambda params, x: network(params, x).real), in_axes=(None, 0)
    )
    df_imag = jax.vmap(
        jax.value_and_grad(lambda params, x: network(params, x).imag), in_axes=(None, 0)
    )

    def loss_prod(grad_logpsi_conj, diff):
        diff = diff.reshape(
            diff.shape + (1,) * (len(grad_logpsi_conj.shape) - len(diff.shape))
        )
        return jnp.nan_to_num(2 * jnp.nanmean(grad_logpsi_conj * diff, axis=0))

    def loss_and_grad(params: ArrayTree, data_and_weights: tuple[jnp.ndarray, jnp.ndarray]):
        data, weights = data_and_weights
        el, other_observables = batch_weighted_local_energy(params, data_and_weights)
        # el = el * weights / jnp.sum(weights)
        # TODO: check if this is correct
        # other_observables = other_observables * weights / jnp.sum(weights)
        pmean_observables = cast(
            OtherObservables,
            jax.tree.map(lambda x: constants.pmean(jnp.mean(x)), other_observables),
        )

        loss = constants.pmean(jnp.nanmean(el))
        clipped_loss = constants.pmean(jnp.nanmean(iqr_clip(el)))
        diff_to_clip = el - clipped_loss
        if system.lz_penalty:
            lz_square = other_observables["angular_momentum_z_square"]
            lz = other_observables["angular_momentum_z"]
            clipped_lz_square = constants.pmean(jnp.nanmean(iqr_clip(lz_square)))
            clipped_lz = constants.pmean(jnp.nanmean(iqr_clip(lz)))
            diff_to_clip += system.lz_penalty * (
                (lz_square - clipped_lz_square)
                - 2 * system.lz_center * (lz - clipped_lz)
            )
        if system.l2_penalty:
            l2 = other_observables["angular_momentum_square"]
            clipped_l2 = constants.pmean(jnp.nanmean(iqr_clip(l2)))
            diff_to_clip += system.l2_penalty * (l2 - clipped_l2)
        diff = iqr_clip(diff_to_clip)

        variance = constants.pmean(jnp.nanmean(el.real**2) - loss.real**2)
        stats = LossStats(**pmean_observables, energy=loss, variance=variance)
        if mode == LossMode.ENERGY_DIFF:
            return stats, diff

        primal_real, tangent_real = df_real(params, data)
        _, tangent_imag = df_imag(params, data)
        kfac_jax.register_normal_predictive_distribution(primal_real[:, None])
        tangent_out = jax.tree.map(
            lambda real, imag: loss_prod(real - 1j * imag, diff),
            tangent_real,
            tangent_imag,
        )

        if mode == LossMode.ENERGY_GRAD:
            return stats, jax.tree.map(jnp.real, tangent_out)
        elif mode == LossMode.SR_F_VECTOR:
            return stats, tangent_out

    return loss_and_grad

def safe_pmean(x, axis_name="qmc_pmap_axis"): #TODO: Check if this is proper
    try:
        return jax.lax.pmean(x, axis_name=axis_name)
    except NameError:
        # no axis bound → just return x
        return x

def make_vvmc_loss_fn(
    network: LogPsiNetwork, system: System, mode: LossMode = LossMode.ENERGY_GRAD
) -> Callable[[ArrayTree, jnp.ndarray], tuple[LossStats, jnp.ndarray]]:
    loss_fn = local_v_energy(network, system)
    batch_local_v_energy = jax.vmap(loss_fn, in_axes=(None, 0))

    df_real = jax.vmap(
        jax.value_and_grad(lambda params, x, dR: jnp.einsum('...ij,...ij->...', 0.5 * (network(params, x).real + network(params, x-dR).real), dR)), in_axes=(None, 0, 0)
    )
    df_imag = jax.vmap(
        jax.value_and_grad(lambda params, x, dR: jnp.einsum('...ij,...ij->...', 0.5 * (network(params, x).imag + network(params, x-dR).imag), dR)), in_axes=(None, 0, 0)
    )

    def loss_prod(grad_logpsi_conj, diff):
        diff = diff.reshape(
            diff.shape + (1,) * (len(grad_logpsi_conj.shape) - len(diff.shape))
        )
        return jnp.nan_to_num(2 * jnp.nanmean(grad_logpsi_conj * diff, axis=0))

    def loss_and_grad(params: ArrayTree, xy_and_dR: tuple[jnp.ndarray, jnp.ndarray]):
        electron_xy, dX = xy_and_dR
        el, other_observables = batch_local_v_energy(params, electron_xy)
        # el = el * weights / jnp.sum(weights)
        # TODO: check if this is correct
        # other_observables = other_observables * weights / jnp.sum(weights)

        pmean_observables = cast(
            OtherObservables,
            jax.tree.map(lambda x: safe_pmean(jnp.mean(x)), other_observables),
        )

        loss = safe_pmean(jnp.nanmean(el)) #TODO: Check if safe_pmean is proper
        clipped_loss = safe_pmean(jnp.nanmean(iqr_clip(el)))
        variance = safe_pmean(jnp.nanmean(el.real**2) - loss.real**2)

        # pmean_observables = cast( 
        #     OtherObservables,
        #     jax.tree.map(lambda x: constants.pmean(jnp.mean(x)), other_observables),
        # )

        # loss = constants.pmean(jnp.nanmean(el))
        # clipped_loss = constants.pmean(jnp.nanmean(iqr_clip(el)))
        diff_to_clip = el - clipped_loss
        diff = iqr_clip(diff_to_clip)

        # variance = constants.pmean(jnp.nanmean(el.real**2) - loss.real**2)
        stats = LossStats(**pmean_observables, energy=loss, variance=variance)
        # if mode == LossMode.ENERGY_DIFF:
        # return stats, diff

        primal_real, tangent_real = df_real(params, electron_xy, dX)
        _, tangent_imag = df_imag(params, electron_xy, dX)
        kfac_jax.register_normal_predictive_distribution(primal_real[:, None])
        tangent_out = jax.tree.map(
            lambda real, imag: loss_prod(real - 1j * imag, diff),
            tangent_real,
            tangent_imag,
        )

        if mode == LossMode.ENERGY_GRAD:
            return stats, jax.tree.map(jnp.real, tangent_out)
        elif mode == LossMode.SR_F_VECTOR:
            return stats, tangent_out

    return loss_and_grad
