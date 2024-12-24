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

import jax
import pytest
from jax import numpy as jnp

from deephall import hamiltonian


def sample(key, batch, nelec):
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def make_lll(nelec: int, Q: int):
    def log_psi(params, data):
        theta, phi = data[..., 0], data[..., 1]
        u = jnp.cos(theta / 2) * jnp.exp(1j * phi / 2)
        v = jnp.sin(theta / 2) * jnp.exp(-1j * phi / 2)
        lll_orb = jnp.stack([u**m * v ** (2 * Q - m) for m in range(nelec)], axis=-1)

        sign, logdet = jnp.linalg.slogdet(lll_orb)
        return logdet + jnp.log(sign)

    return log_psi


def test_free_electron():
    def log_psi(params, data):
        """Sphere harmonics with l=1: $Y_{1m}$."""
        theta, phi = data[..., 0], data[..., 1]
        orb = jnp.stack(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.cos(theta),
                jnp.sin(theta) * jnp.sin(phi),
            ],
            axis=-1,
        )
        sign, logdet = jnp.linalg.slogdet(orb)
        return logdet + jnp.log(sign)

    data = sample(jax.random.PRNGKey(1898), 2, nelec=3)
    laplacian = hamiltonian.make_local_kinetic_energy(log_psi, Q=0, r=1)
    batch_laplacian = jax.jit(jax.vmap(laplacian, in_axes=(None, 0)))
    ke, other_observables = batch_laplacian(None, data)
    assert jnp.allclose(ke, 3, atol=1e-3)
    assert jnp.allclose(other_observables["angular_momentum_square"], 0, atol=1e-3)


@pytest.mark.parametrize("nelec,Q,L_square", [(1, 1, 2), (3, 1, 0), (9, 4, 0)])
def test_kinetic_and_angular_momentum(nelec: int, Q: int, L_square: float):
    data = sample(jax.random.PRNGKey(1898), 2, nelec)
    laplacian = hamiltonian.make_local_kinetic_energy(
        make_lll(nelec, Q), Q, jnp.sqrt(Q)
    )
    batch_laplacian = jax.jit(jax.vmap(laplacian, in_axes=(None, 0)))
    ke, other_observables = batch_laplacian(None, data)
    assert jnp.allclose(ke, nelec / 2, atol=1e-3)
    assert jnp.allclose(
        other_observables["angular_momentum_square"], L_square, atol=1e-3
    )
