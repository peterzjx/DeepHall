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


class MLP(nn.Module):
    features: tuple[int] = (16, 16, 2)

    @nn.compact
    def __call__(self, z_Z):
        assert self.features[-1] == 2
        z = z_Z[0] - z_Z[1]
        x = jnp.stack([z.real, z.imag])
        for feat in self.features[:-1]:
            x = nn.sigmoid(nn.Dense(feat)(x))
        out = nn.Dense(self.features[-1])(x)

        d_real = out[0]
        d_imag = out[1]

        d = d_real + 1j * d_imag
        return d

class DipoleLaughlin(nn.Module):
    """Create Laughlin wavefunction for ground or quasiparticle/quasihole state."""
    features: tuple[int]
    nspins: tuple[int, int]
    flux: float
    cf_flux: int = 1
    "Flux p for composite fermion."

    
    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - self.cf_flux * (sum(self.nspins) - 1)
        self.dipole_vector = MLP(self.features)
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
        ln_laughlin = ln_wfn0 + ln_wfn1
        
        zizj, z_Z = extract_triplets(complex_z)

        rij2 = (zizj[..., 0] - zizj[..., 1])**2
        print("rij2", rij2.shape, z_Z.shape)
        get_dij = jax.vmap(self.dipole_vector, in_axes = 0)
        dij = get_dij(z_Z)
        print("rij2", dij.shape)
        dipole_correction = jnp.sum(jnp.log(dij**2 / rij2),axis=-1)
        print("dipole_correction", dipole_correction.shape, ln_laughlin.shape)


        return ln_laughlin + dipole_correction
