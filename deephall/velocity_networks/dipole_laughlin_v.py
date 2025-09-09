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
from deephall.velocity_networks.attention_block import PairSystemFeatureNet

def extract_electron_pairs(electron):
    Ne = electron.shape[0]
    # indices for anchors and partners
    anchors = jnp.arange(Ne)[:, None]                  # shape [Ne, 1]
    partners = (anchors + jnp.arange(1, Ne)) % Ne      # shape [Ne, Ne-1]

    # gather coordinates
    z_anchor = electron[anchors]       # shape [Ne, 1, 2]
    z_partner = electron[partners]     # shape [Ne, Ne-1, 2]

    # broadcast to build pairs
    pairs = jnp.stack([jnp.broadcast_to(z_anchor, z_partner.shape),
                      z_partner], axis=-2)  # shape [Ne, Ne-1, 2, 2]
    return pairs
class AttentionVelocityNet(nn.Module):
    d_model: int = 32
    d_mlp_model: int = 16
    num_heads: int = 2
    num_layers: int = 1   # number of attention-fusion blocks
    num_MLP_layers: int = 3 # number of attention-fusion blocks    
    def setup(self):
        self.feat_net_re = PairSystemFeatureNet(self.d_model, self.num_heads, self.num_layers)
        self.feat_net_im = PairSystemFeatureNet(self.d_model, self.num_heads, self.num_layers)
    @nn.compact
    def __call__(self, cplx_zij, cplx_env):
        zij = jnp.stack([jnp.real(cplx_zij), jnp.imag(cplx_zij)], axis=-1)
        zi, zj = zij[..., 0], zij[..., 1]
        env = jnp.stack([jnp.real(cplx_env), jnp.imag(cplx_env)], axis=-1)
        feature_re = self.feat_net_re(zij, env)
        feature_im = self.feat_net_im(zij, env)

        for i in range(self.num_MLP_layers):
            feature_re = nn.Dense(self.d_mlp_model)(feature_re)
            feature_re = nn.sigmoid(feature_re)   # or relu/sigmoid/tanh
        velocity_re = nn.Dense(2)(feature_re)

        for i in range(self.num_MLP_layers):
            feature_im = nn.Dense(self.d_mlp_model)(feature_im)
            feature_im = nn.sigmoid(feature_im)   # or relu/sigmoid/tanh
        velocity_im = nn.Dense(2)(feature_im)
        velocity = (velocity_re + 1j * velocity_im) * jnp.exp(-0.02 * (jnp.abs(zi)**2 + jnp.abs(zj)**2))
        return velocity


class DipoleLaughlinVelocity(nn.Module):
    """Create drift velocity for the Laughlin wavefunction."""
    nspins: tuple[int, int]
    flux: float
    hidden_features: tuple[int] = (64, 64)  # e.g., [64, 64]
    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - (nelec - 1)
        # self.features = self.hidden_features +(4,)
        self.get_pairwise_attention = AttentionVelocityNet()
        # self.ManyBodyV = ManyBodyVelocity(self.features)
        # assert self.features[-1] == 4
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
        env_rolled = jnp.stack([jnp.roll(electrons_xy, shift=-k, axis=0) for k in range(Ne)], axis=0)
        env_stack = jnp.tile(env_rolled[:, None, :, :], (1, Ne-1, 1, 1)) 
        # z_stack = jnp.tile(electrons_xy[None, None, :, :], (Ne, Ne-1, 1, 1)) 
        # print("xxx shape",env_rolled.shape, env_stack.shape)
        # print(env_rolled)
        cpx_zij_pairs = zij_pairs[..., 0] + 1j * zij_pairs[..., 1]
        cpx_env_stack = env_stack[..., 0] + 1j * env_stack[..., 1]
        # print('zij_pair shape', electrons_xy.shape, zij_pairs.shape, cpx_zij_pairs.shape, cpx_z_stack.shape)
        
        get_vij = jax.vmap(
            jax.vmap(self.get_pairwise_attention, in_axes=(0, 0)),  # inner vmap over axis=0
            in_axes=(0, 0)                                          # outer vmap over axis=0
        )
        cpx_vij = get_vij(cpx_zij_pairs,  cpx_env_stack)  # shape (N, N-1, 2)
        # print('cpx_vij', cpx_vij[...,0])
        atten_v2 = jnp.sum(cpx_vij, axis = (1,))
        # print('cpx_vij shape', cpx_vij.shape, atten_v2.shape, atten_v2.dtype)
        # jax.debug.print("attention v = {}", atten_v2[0])
        drift_v = v1 + 1 * integer_v2  +  2 * atten_v2
        return drift_v
