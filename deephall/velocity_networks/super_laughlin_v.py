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


def extract_rotating_features(xy):
    """
    xy: [..., Ne, 2]
    returns: [..., Ne, Ne, 2]
    """
    Ne = xy.shape[-2]
    idx = (jnp.arange(Ne)[None, :] + jnp.arange(Ne)[:, None]) % Ne
    # idx: [Ne, Ne], each row is a rotation
    return xy[..., idx, :] 

class MLP(nn.Module):
    features: tuple[int]

    @nn.compact
    def __call__(self, x):
        assert self.features[-1] == 4
        x = x.flatten()  # [2, 2] -> [4] for a single sample
        for feat in self.features[:-1]:
            x = nn.sigmoid(nn.Dense(feat)(x))
        out = nn.Dense(self.features[-1])(x)

        vx_real = out[0]
        vx_imag = out[1]
        vy_real = out[2]
        vy_imag = out[3]

        vx = vx_real + 1j * vx_imag
        vy = vy_real + 1j * vy_imag
        return jnp.array([vx, vy])

class SmoothMLP(nn.Module):
    features: tuple[int]
    sigma: float = 0.05  # Gaussian kernel width
    n_samples: int = 8   # Number of smoothing samples

    def setup(self):
        self.mlp = MLP(self.features)

    def mollify(self, x):
        """
        Apply Gaussian smoothing to the MLP output over a local neighborhood.
        """
        key = self.make_rng('mollify')
        # Sample perturbations in input space
        perturbations = jax.random.normal(key, (self.n_samples,) + x.shape) * self.sigma
        # Shift inputs
        neighbors = x + perturbations
        # Evaluate raw MLP at each neighbor
        vals = jax.vmap(self.mlp)(neighbors)
        # Gaussian weights
        weights = jnp.exp(-jnp.sum(perturbations**2, axis=1) / (2 * self.sigma**2))
        weights /= jnp.sum(weights)
        # Weighted average
        return jnp.tensordot(weights, vals, axes=1)

    def __call__(self, x):
        raw_output = self.mlp(x)
        smooth_output = self.mollify(x)
        return smooth_output

class TwoBodyVelocity(nn.Module):
    features: tuple[int]  # e.g., [64, 64, 1]

    @nn.compact
    def __call__(self, z):
        assert z.shape == (2 , 2)
        assert self.features[-1] == 4
    ########### 1/3 Laughlin original form #######################
        z1 = z[0][0] + 1j * z[0][1]
        z2 = z[1][0] + 1j * z[1][1]
        vx = 1.0 / (z1 - z2)
        vy = 1j / (z1 - z2)
        v_laughlin = 3 * jnp.stack([vx, vy], axis = -1)
    ######################################
        g = MLP(self.features) 
        return v_laughlin * (1 + g(z) * jnp.exp(-0.01 * (jnp.abs(z1)**2 + jnp.abs(z2)**2)))

class ManyBodyVelocity(nn.Module):
    features: tuple[int]  # e.g., [64, 64, 1]

    @nn.compact
    def __call__(self, z):
        assert self.features[-1] == 4
        v = MLP(self.features) 
        return 0.1 * v(z) * jnp.exp(-0.001 * jnp.sum(z[...,0]**2 + z[...,1]**2))

class SuperLaughlinVelocity(nn.Module):
    """Create drift velocity for the Laughlin wavefunction."""
    nspins: tuple[int, int]
    flux: float
    hidden_features: tuple[int] = (16, 16)  # e.g., [64, 64]
    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - (nelec - 1)
        self.features = self.hidden_features +(4,)
        self.TwoBodyV = TwoBodyVelocity(self.features) 
        # self.ManyBodyV = ManyBodyVelocity(self.features)
        assert self.features[-1] == 4
        assert nelec == 2 * self.Q1 + 1  # Ground state for 1/3

    def __call__(self, electrons_xy):
        print('##xy shape', electrons_xy.shape)
        Ne = sum(self.nspins)
        x, y = electrons_xy[..., 0, None], electrons_xy[..., 1, None]
        r = jnp.sqrt(x**2 + y**2)

        vx = 3 * (1 - Ne) / (1 + r**2) * x
        vy = 3 * (1 - Ne) / (1 + r**2) * y 

        v1 = jnp.concatenate([vx, vy], axis = -1)
        
        electrons_pairs = extract_electron_pairs(electrons_xy) #[Ne, Ne-1, 2, 2]
        batched_v2 = jax.vmap(             # over i (Ne)
            jax.vmap(self.TwoBodyV, in_axes=0),       # over j (Ne-1)
            in_axes=0
        )
        v2_map = batched_v2(electrons_pairs) #[Ne, Ne-1, 2]
        v2 = jnp.sum(v2_map, axis = -2)

        # rotating_features = extract_rotating_features(electrons_xy)
        # batched_v_many = jax.vmap(
        #     jax.vmap(self.ManyBodyV, in_axes=0),
        #     in_axes=0
        # )
        # v_many_map = batched_v_many(rotating_features)
        # v_many = jnp.sum(v_many_map, axis = -2)
        drift_v = v1 + v2
        # drift_v = drift_v + v_many
        return drift_v
