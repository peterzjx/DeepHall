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

from flax import linen as nn
from jax import numpy as jnp
import jax

def antisym_matrix(u, v):
    """
    u, v: [Ne]
    return: [Ne, Ne] with M_ij = u_i v_j - u_j v_i
    """
    ui, uj = u[:, None], u[None, :]
    vi, vj = v[:, None], v[None, :]
    return ui * vj - uj * vi

def extract_triplets(z):
    """
    z: [Ne] complex coordinates
    returns: [Ne*(Ne-1)//2, 3], each row = [zi, zj, Z_rest]
    """
    Ne = z.shape[0]
    idx_i, idx_j = jnp.triu_indices(Ne, k=1)  # all i<j pairs
    
    zi = z[idx_i]
    zj = z[idx_j]
    
    # full sum of all coordinates
    total = jnp.sum(z)
    
    # Z_rest = (sum of all except i,j) / (Ne-2)
    Z = (total - zi - zj) / (Ne - 2)
    
    # stack into [N_pairs, 3]
    zizj = jnp.stack([zi, zj], axis=-1)
    z_Z = jnp.stack([(zi+zj) / 2, Z], axis=-1)
    return zizj, z_Z

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

    
class RelativeAttention(nn.Module):
    num_heads: int = 2
    hidden_dim: int = 16
    max_disp: float = 0.08
    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        """
        inputs: shape (2,) complex array [z, Z]
        output: complex displacement d = dx + 1j * dy
        """
        # Step 1: split into real & imag → shape (2, 2)
        coords = jnp.stack([inputs.real, inputs.imag], axis=-1)

        # Step 2: encode each particle
        x = nn.Dense(self.hidden_dim)(coords)

        # Step 3: self-attention over two particles
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim
        )(x[None, :, :])  # (1, 2, hidden_dim)

        attn_out = attn_out[0]  # (2, hidden_dim)

        # Step 4: relative feature
        rel_feature = attn_out[0] - attn_out[1]  # (hidden_dim,)

        # Step 5: learn raw displacement
        disp_raw = nn.Dense(2)(rel_feature)

        # Step 6: learn gate
        gate = nn.sigmoid(nn.Dense(1)(rel_feature))  # scalar in (0,1)

        # Step 7: learnable scale, confined to [0, 0.05]
        raw_scale = self.param("raw_scale", nn.initializers.constant(0.5), (1,))
        truncate = self.max_disp * nn.sigmoid(raw_scale)  # always ≤ 0.05

        # Step 8: final displacement
        disp = truncate * gate.flatten() * jnp.tanh(disp_raw)

        # Step 9: complex output
        d = disp[0] + 1j * disp[1]
        return d

        
class MLP(nn.Module):
    features: tuple[int] = (32, 32, 2)

    @nn.compact
    def __call__(self, z_Z):
        # assert self.features[-1] == 2
        # z = z_Z[0] - z_Z[1]
        # x = jnp.stack([z.real, z.imag])
        # for feat in self.features[:-1]:
        #     x = nn.sigmoid(nn.Dense(feat)(x))
        # out = nn.Dense(self.features[-1])(x)

        # d_real = out[0]
        # d_imag = out[1]

        # d = (d_real + 1j * d_imag)  * 0.0001
        # return d
        direction = z_Z[1] - z_Z[0]
        direction = direction / jnp.abs(direction)
        z = z_Z[0] - z_Z[1]
        x = jnp.stack([z.real, z.imag])
        for feat in self.features[:-1]:
            x = nn.sigmoid(nn.Dense(feat)(x))
        out = nn.Dense(self.features[-1])(x)
        rho = jnp.sum(out) * 0.02
        result = direction * rho
        print('h_one', result.shape)
        return result

# class EnvAttention(nn.Module):
#     hidden_dim: int = 64
#     num_heads: int = 4

#     @nn.compact
#     def __call__(self, env_coords):
#         # env_coords: shape [N_env], complex
#         x = jnp.stack([env_coords.real, env_coords.imag], axis=-1)  # [N_env, 2]
#         # simple self-attention to aggregate environment
#         attn = nn.SelfAttention(
#             num_heads=self.num_heads,
#             qkv_features=self.hidden_dim,
#             out_features=self.hidden_dim
#         )(x[None])  # add batch dim
#         env_embed = attn.mean(axis=1)  # [hidden_dim]
#         return jnp.squeeze(env_embed)
    
class PairDisplacementNet(nn.Module):
    max_disp: float = 0.05   # tunable but bounded
    d_model: int = 16        # hidden size, pair_wise_feat & env_feat dim
    num_heads: int = 4

    @nn.compact
    def __call__(self, zij, E):
        """
        zij: complex pair [zi, zj]
        E: 1D array of complex environment electrons
        """

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
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model
        )(query, env_feat[None, :, :], env_feat[None, :, :])
        attn = jnp.squeeze(attn)  # (1, d_model) → (d_model,)
        # ---- 3. Fuse pair + attention ----
        fused = jnp.concatenate([pair_feat, attn])
        fused = nn.sigmoid(nn.Dense(self.d_model)(fused))
        print('attn', attn.shape, pair_feat.shape, fused.shape)
        # ---- 4. Predict displacement ----
        out = nn.Dense(2)(fused)  # (dx, dy)
        out = jnp.tanh(out) * self.max_disp

        d = out[0] + 1j * out[1]
        # Return complex displacement
        return d

class DipoleLaughlin(nn.Module):
    """Create Laughlin wavefunction for ground or quasiparticle/quasihole state."""
    features: tuple[int]
    nspins: tuple[int, int]
    flux: float
    cf_flux: int = 1
    "Flux p for composite fermion."
    dipole_mode = "pair_env_attention"

    
    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - self.cf_flux * (sum(self.nspins) - 1)
        
        if self.dipole_mode == "simple_mlp":
            self.dipole_vector = MLP(self.features)
        elif self.dipole_mode == "simple_attention":
            self.get_pairwise_attention = RelativeAttention()
        elif self.dipole_mode == "pair_env_attention":
            # self.get_env_electron_feat = EnvAttention() # Taking all N-2 electrons as environemt and extract features
            self.get_pairwise_attention = PairDisplacementNet()
        # self.dipole_vector = MLP(self.features)
        # self.get_pairwise_attention = RelativeAttention()
        if nelec == 2 * self.Q1 + 1:  # Ground state
            pass
        else:
            raise ValueError("Filling not supported")
    @nn.compact
    def __call__(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))
################## Laughlin wavefunction in Haldane spinor ##################        
        # uivj = antisym_matrix(u, v)
        # element = uivj + jnp.eye(uivj.shape[0])
        # jastrow = jnp.prod(element)
        # ln_laughlin = 3 / 2 * jnp.log(jastrow)
#############################################################################
        complex_z = u / v
        complex_zij = complex_z[..., :, None] -  complex_z[..., None, :] 
        masked_complex_zij = complex_zij + jnp.eye(complex_zij.shape[0])
        
        # ln_wfn0 = - self.flux / 2 * jnp.sum(jnp.log(1 + jnp.abs(complex_z)**2), axis = -1) #Wang-Yang Gauge
        ln_wfn0 = - self.flux / 2 * jnp.sum(jnp.log((1 /jnp.abs(complex_z) + jnp.abs(complex_z))*complex_z), axis = -1) #Haldane Gauge
        
        ln_wfn1 = 3 / 2 * jnp.sum(jnp.log(masked_complex_zij))
        
############ simple dipole correction ########################################
        # zizj, z_Z = extract_triplets(complex_z)
        # rij2 = (zizj[..., 0] - zizj[..., 1])**2
        # # get_dij = jax.vmap(self.dipole_vector, in_axes = 0)
        # get_dij = jax.vmap(self.get_pairwise_attention, in_axes = 0)
        # dij = get_dij(z_Z)
        # dipole_correction =  jnp.sum(jnp.log(1-dij**2 / rij2),axis=-1)
############ simple dipole correction ########################################
        _, zij_pairs, envs = extract_zizj_env(complex_z)
        zij_pairs_both = jnp.stack([zij_pairs, zij_pairs[:, ::-1]], axis=1)  # shape (N, 2, ...)
        envs_both = jnp.stack([envs, envs], axis=1)  # shape (N, 2, ...)
        get_dij = jax.vmap(
            jax.vmap(self.get_pairwise_attention, in_axes=(0, 0)),  # inner vmap over axis=1
            in_axes=(0, 0)                                          # outer vmap over axis=0
        )
        dij_both = get_dij(zij_pairs_both, envs_both)  # shape (N, 2, ...)
        dij = jnp.sum(dij_both, axis=1) * 0.5
        rij2 = (zij_pairs[..., 0] - zij_pairs[..., 1])**2
        dipole_correction =  jnp.sum(jnp.log(1-dij**2 / rij2),axis=-1)
        ln_laughlin = ln_wfn0 + ln_wfn1 + dipole_correction

        return ln_laughlin
