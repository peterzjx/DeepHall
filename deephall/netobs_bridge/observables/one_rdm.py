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

from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from netobs.observables import Estimator, Observable
from scipy import special as ss

from deephall.netobs_bridge.hall_system import HallSystem


class OneRDM(Observable[HallSystem]):
    def shapeof(self, system) -> tuple[int, ...]:
        norbs = system["flux"] + 1
        return (norbs, norbs)


def make_monopole_harm(q, l, m):  # NOQA
    norm_factor = np.sqrt(
        ((2 * l + 1) / (4 * np.pi))
        * (ss.factorial(l - m) * ss.factorial(l + m))
        / (ss.factorial(l - q) * ss.factorial(l + q))
    )
    s = np.arange(l - m + 1)
    sum_factors = jnp.array(
        (-1) ** (l - m - s) * ss.comb(l - q, s) * ss.comb(l + q, l - m - s)
    )

    def Y_qlm(electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        # Clip for stability
        x = jnp.clip(jnp.cos(theta), -1 + 1e-4, 1 - 1e-4)
        theta_part = jnp.sum(
            sum_factors
            * (1 - x[..., None]) ** (l - s - (m + q) / 2)
            * (1 + x[..., None]) ** (s + (m + q) / 2),
            axis=-1,
        )
        return norm_factor / 2**l * theta_part * jnp.exp(1j * m * phi)

    return Y_qlm


def uniform_sample(key, batch):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, batch, minval=-1, maxval=1))
    phi = jax.random.uniform(key2, batch, minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


class OneRDMEstimator(Estimator[HallSystem]):
    observable_type = OneRDM

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.batch_network = jax.vmap(adaptor.call_network, in_axes=(None, 0, None))
        Q = system["flux"] / 2
        self.orbitals = [make_monopole_harm(Q, Q, m) for m in np.arange(-Q, Q + 1)]
        self.batch_product = jax.pmap(
            jax.vmap(self.eval_product, in_axes=(None, 0, None, 0)),
            in_axes=(0, 0, None, 0),
        )

    def empty_val_state(
        self, steps: int
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        empty_values = {
            "one_rdm": jnp.zeros(
                (steps, *self.observable.shape), self.options.get("dtype", "complex64")
            ),
        }
        return empty_values, {}

    def eval_product(self, params, data, system, r_prime):
        nelec = len(data)
        data_prime = jnp.repeat(data[None], nelec, axis=0)
        data_prime = data_prime.at[*jnp.diag_indices(nelec, 2)].set(r_prime)

        logpsi = self.adaptor.call_network(params, data, system)
        logpsi_prime = self.batch_network(params, data_prime, system)
        varphi = jnp.stack([orb(data) for orb in self.orbitals], axis=-1)
        varphi_prime = jnp.stack([orb(r_prime) for orb in self.orbitals], axis=-1)

        wf_ratio = jnp.exp(logpsi_prime - logpsi[..., None])

        # math: < \sum_a{ Psi^*(Ra')/Psi^*(R) * phi_i(ra) phi_j*(ra') } >
        return (4 * jnp.pi) * jnp.sum(
            wf_ratio[..., None, None]
            * varphi[..., None]
            * jnp.conj(varphi_prime)[..., None, :],
            axis=0,
        )

    def evaluate(
        self, i, params, key, data, system, state, aux_data
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del i, aux_data
        r_prime = uniform_sample(key[0], data.shape[:2])[:, :, None, :]
        product = self.batch_product(params, data, system, r_prime)
        return {"one_rdm": product}, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state
        one_rdm = jnp.mean(all_values["one_rdm"], axis=0)
        return {"diagonal": jnp.diagonal(one_rdm), "trace": jnp.trace(one_rdm)}


DEFAULT = OneRDMEstimator  # Useful in CLI
