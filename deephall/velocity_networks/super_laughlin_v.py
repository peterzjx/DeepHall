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
from flax import linen as nn
from jax import numpy as jnp


def extract_electron_pairs(electron):
    Ne = electron.shape[0]
    
    # Create all possible pairs (including self-pairs)
    ei = jnp.repeat(electron[:, None, :], Ne, axis=1)  # shape: [Ne, Ne, 2]
    ej = jnp.repeat(electron[None, :, :], Ne, axis=0)  # shape: [Ne, Ne, 2]

    # Create indices for non-diagonal elements
    # For each electron i, we want pairs with all electrons j != i
    # We can do this by creating a list of indices for each i
    indices = []
    for i in range(Ne):
        # For electron i, get all j != i
        j_indices = jnp.concatenate([jnp.arange(i), jnp.arange(i+1, Ne)])
        indices.append(j_indices)
    
    # Stack the indices for all electrons
    all_indices = jnp.stack(indices)  # shape: [Ne, Ne-1]
    
    # Use advanced indexing to get the pairs
    # For each electron i, get the pairs with electrons j != i
    ei_pairs = jnp.take_along_axis(ei, all_indices[:, :, None], axis=1)  # shape: [Ne, Ne-1, 2]
    ej_pairs = jnp.take_along_axis(ej, all_indices[:, :, None], axis=1)  # shape: [Ne, Ne-1, 2]
    
    # Stack the pairs along the last axis
    pairs = jnp.stack([ei_pairs, ej_pairs], axis=-2)  # shape: [Ne, Ne-1, 2, 2]
    
    return pairs  # shape: [Ne, Ne-1, 4]


class MLP(nn.Module):
    features: tuple[int]

    @nn.compact
    def __call__(self, x):
        assert self.features[-1] == 4
        x = x.flatten()  # flatten [2, 2] -> [4]
        for feat in self.features[:-1]:
            x = nn.sigmoid(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        vx_real = x[0]
        vx_imag = x[1]
        vy_real = x[3]
        vy_imag = x[4]
        vx = vx_real + 1j * vx_imag 
        vy = vy_real + 1j *vy_imag
        return jnp.array([vx, vy])

class TwoBodyVelocity(nn.Module):
    features: tuple[int]  # e.g., [64, 64, 1]

    @nn.compact
    def __call__(self, z):
        assert self.features[-1] == 4
        g = MLP(self.features)
        ex_z = jnp.flip(z, axis=0)  # Swap z1 and z2: [z2, z1]
        return g(z) - g(ex_z)

class SuperLaughlinVelocity(nn.Module):
    """Create drift velocity for the Laughlin wavefunction."""
    nspins: tuple[int, int]
    flux: float
    hidden_features: tuple[int] = (32, 32)  # e.g., [64, 64]
    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - (nelec - 1)
        self.features = self.hidden_features +(4,)
        self.TwoBodyV = TwoBodyVelocity(self.features) 
        assert self.features[-1] == 4
        assert nelec == 2 * self.Q1 + 1  # Ground state for 1/3

    def __call__(self, electrons_xy):
        Ne = sum(self.nspins)
        x, y = electrons_xy[..., 0, None], electrons_xy[..., 1, None]
        r = jnp.sqrt(x**2 + y**2)

        vx = 3 * (1 - Ne) / (1 + r**2) * x
        vy = 3 * (1 - Ne) / (1 + r**2) * y 

        v1 = jnp.concatenate([vx, vy], axis = -1)
        
        electrons_pairs = extract_electron_pairs(electrons_xy)
        batched_velocity = jax.vmap(             # over i (Ne)
            jax.vmap(self.TwoBodyV, in_axes=0),       # over j (Ne-1)
            in_axes=0
        )
        pair_v = batched_velocity(electrons_pairs) #[Ne, Ne-1, 2]
        
        # assert pair_v.shape == (Ne, Ne-1, 2)
        v2 = jnp.sum(pair_v, axis = -2)
        # print('pair_v shape', pair_v.shape, electrons_pairs.shape, v2.shape)
        # assert v2.shape == (Ne, 2) #complex64
        print('vshapes', v1.shape, v2.shape)
        drift_v = v1 + v2
        return drift_v
