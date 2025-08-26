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

def extract_zizj_env(z: jnp.ndarray):
    """
    Extract pairwise features in a JAX-friendly way.

    Args:
        z: jnp.ndarray of shape (N,), complex array
    
    Returns:
        pair_idx: (num_pairs, 2) int32 array with all pairs (i, j), i < j
        pairs: (num_pairs, 2) complex array, containing (zi, zj)
        env: (num_pairs, N-2) complex array, environment for each pair
    """
    N = z.shape[0]
    # Precompute pair indices (static for a given N)
    idx_i, idx_j = jnp.triu_indices(N, k=1)   # i < j
    num_pairs = idx_i.shape[0]

    # Gather pairs
    zi = z[idx_i]
    zj = z[idx_j]
    pairs = jnp.stack([zi, zj], axis=-1)   # (num_pairs, 2)
    env = jnp.repeat(z[None, :], num_pairs, axis=0)  # shape: (num_pairs, N)
    pair_idx = jnp.stack([idx_i, idx_j], axis=-1)  # (num_pairs, 2)
    return pair_idx, pairs, env

class AttentionVelocityNet(nn.Module):
    d_model: int = 16        # hidden size, pair_wise_feat & env_feat dim
    num_heads: int = 4

    @nn.compact
    def __call__(self, zij, E):
        """
        zij: complex pair [zi, zj]
        E: 1D array of complex environment electrons
        """
        Ne = E.shape[0]
        # ---- 1. Encode zi, zj ----
        zi, zj = zij[..., 0], zij[..., 1]
        pair_feat = jnp.stack([
            jnp.real(zi), jnp.imag(zi),
            jnp.real(zj), jnp.imag(zj),
            jnp.real(zi - zj), jnp.imag(zi - zj),
        ])  # shape (6,)

        pair_feat = nn.Dense(self.d_model)(pair_feat)

        # ---- 2. Encode environment ----
        env_feat = jnp.stack([jnp.real(E), jnp.imag(E)], axis=-1)  # (Ne-2, 2)
        env_feat = nn.Dense(self.d_model)(env_feat)

        # Apply attention: query = pair, key/value = environment
        query = pair_feat[None, None, :]      # shape (1, 1, d_model)

        attn_re = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model
        )(query, env_feat[None, :, :], env_feat[None, :, :])
        attn_re = jnp.squeeze(attn_re)  # (1, d_model) → (d_model,)
        # ---- 3. Fuse pair + attention ----
        fused_re = jnp.concatenate([pair_feat, attn_re])
        fused_re = nn.sigmoid(nn.Dense(self.d_model)(fused_re))
        # ---- 4. Predict displacement ----
        out = nn.Dense(Ne * 2)(fused_re)  
        velocity_re = jnp.tanh(out)
        velocity_re = velocity_re.reshape([Ne, 2])

        attn_im = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model
        )(query, env_feat[None, :, :], env_feat[None, :, :])
        attn_im = jnp.squeeze(attn_im)  # (1, d_model) → (d_model,)
        # ---- 3. Fuse pair + attention ----
        fused_im = jnp.concatenate([pair_feat, attn_im])
        fused_im = nn.sigmoid(nn.Dense(self.d_model)(fused_im))
        # ---- 4. Predict displacement ----
        out = nn.Dense(Ne * 2)(fused_im)
        velocity_im = jnp.tanh(out)
        velocity_im = velocity_re.reshape([Ne, 2])

        velocity = velocity_re + 1j * velocity_im # [[v1x, v1y], [v2x, v2y],...,[vNx, vNy]]~[Ne, 2], complex value
        print('velocity shape', velocity.shape)
        return velocity

class DipoleLaughlinVelocity(nn.Module):
    """Create drift velocity for the Laughlin wavefunction."""
    nspins: tuple[int, int]
    flux: float
    hidden_features: tuple[int] = (16, 16)  # e.g., [64, 64]
    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - (nelec - 1)
        self.features = self.hidden_features +(4,)
        self.get_pairwise_attention = AttentionVelocityNet()
        # self.ManyBodyV = ManyBodyVelocity(self.features)
        assert self.features[-1] == 4
        assert nelec == 2 * self.Q1 + 1  # Ground state for 1/3
    def calc_laughlin_pair_v(self, electrons_xy):
        Ne = sum(self.nspins)
        x, y = electrons_xy[..., 0, None], electrons_xy[..., 1, None]
        xi = x[:, None, :]
        xj = x[None, :, :]
        xij = xi-xj

        yi = y[:, None, :]
        yj = y[None, :, :]
        yij = yi-yj

        zij = jnp.concatenate([xij, yij], axis = -1)
        x_zij = jnp.concatenate([-yij, xij], axis = -1)
        rij2 = (xij**2 + yij**2) + 1e-10

        weights = 1.0 / rij2  # shape: (N, N, 1)
        mask = ~jnp.eye(Ne, dtype=bool)  # shape: (N, N)
        mask = mask[..., None]
        weights = weights * mask  # zero out diagonal

        weighted_zij = zij * weights  # shape: (N, N, 2)
        iweighted_x_zij = x_zij * weights

        v_paired = jnp.sum(weighted_zij, axis=1)  # shape: (N, 2)
        iv_paired = jnp.sum(iweighted_x_zij, axis=1)  # shape: (N, 2)

        return v_paired + 1j*iv_paired
    
    def __call__(self, electrons_xy):
        Ne = sum(self.nspins)
        x, y = electrons_xy[..., 0, None], electrons_xy[..., 1, None]
        # complex_z = electrons_xy[..., 0] + 1j * electrons_xy[..., 1]
        r = jnp.sqrt(x**2 + y**2)

        vx = 3 * (1 - Ne) / (1 + r**2) * x
        vy = 3 * (1 - Ne) / (1 + r**2) * y 
        v1 = jnp.concatenate([vx, vy], axis = -1)        
        integer_v2 = self.calc_laughlin_pair_v(electrons_xy) #corresponding to \sum_{i,j}(zi-zj) in IQHE wfn
        

        zij_pairs = extract_electron_pairs(electrons_xy)
        z_stack = jnp.tile(electrons_xy[None, None, :, :], (Ne, Ne-1, 1, 1)) 

        cpx_zij_pairs = zij_pairs[..., 0] + 1j * zij_pairs[..., 1]
        cpx_z_stack = z_stack[..., 0] + 1j * z_stack[..., 1]
        print('zij_pair shape', electrons_xy.shape, zij_pairs.shape, cpx_zij_pairs.shape, cpx_z_stack.shape)
        
        # get_vij = jax.vmap(
        #     jax.vmap(self.get_pairwise_attention, in_axes=(0, 0)),  # inner vmap over axis=0
        #     in_axes=(0, 0)                                          # outer vmap over axis=0
        # )
        # cpx_vij_both = get_vij(cpx_zij_pairs, cpx_z_stack)  # shape (N, 2, ...)
        
        # atten_v2 = jnp.sum(cpx_vij_both, axis=(0,1))
        # print('cpx_vij_both shape', cpx_vij_both.shape, atten_v2.shape)
        drift_v = v1 + 3 * integer_v2 # + atten_v2
        return drift_v
