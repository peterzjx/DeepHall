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

"""This code is an reimplementation of the Psiformer network (Glehn et al., ICLR 2023).

The input feature is chosen as the Cartesian coordinates of the elelctrons:
    h_one_0 = [cos(theta), sin(theta) cos(phi), sin(theta) sin(phi)]
The feature is then passed through standard Psiformer layers, outputing features h_one.
Afterwards, the features are used to construct the orbitals with the monopole harmonics.
The details for the orbital construction are located in `blocks.py`.
"""

from flax import linen as nn
from jax import numpy as jnp

from deephall.config import OrbitalType

from .blocks import Jastrow, Orbitals
# import jax
# jax.config.update("jax_debug_nans", True)

class PsiformerLayers(nn.Module):
    num_heads: int
    heads_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, electrons: jnp.ndarray, spins: jnp.ndarray):
        theta, phi = electrons[..., 0], electrons[..., 1]
        h_one = self.input_feature(theta, phi, spins)
        attention_dim = self.num_heads * self.heads_dim
        h_one = nn.Dense(attention_dim, use_bias=False)(h_one)
        for _ in range(self.num_layers):
            attn_out = nn.MultiHeadAttention(num_heads=self.num_heads)(h_one)
            h_one += nn.Dense(attention_dim, use_bias=False)(attn_out)
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
            h_one += nn.tanh(nn.Dense(attention_dim)(h_one))
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
        return h_one

    def input_feature(self, theta: jnp.ndarray, phi: jnp.ndarray, spins: jnp.ndarray):
        return jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                spins,
            ],
            axis=-1,
        )


class Psiformer(nn.Module):
    nspins: tuple[int, int]
    Q: float
    ndets: int
    num_heads: int
    heads_dim: int
    num_layers: int
    orbital_type: OrbitalType

    def __call__(self, electrons):
        orbitals = self.orbitals(electrons)
        
        signs, logdets = jnp.linalg.slogdet(orbitals)
        print(orbitals.shape, logdets.shape)
        logmax = jnp.max(logdets)  # logsumexp trick
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

    @nn.compact
    def orbitals(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        spins = jnp.array([1] * self.nspins[0] + [-1] * self.nspins[1])
        h_one = PsiformerLayers(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            heads_dim=self.heads_dim,
        )(electrons, spins)
        orbitals = Orbitals(
            type=self.orbital_type, Q=self.Q, nspins=self.nspins, ndets=self.ndets
        )(h_one, theta, phi)
        jastrow = Jastrow(self.nspins)(electrons)
        return jnp.exp(jastrow / sum(self.nspins)) * orbitals
