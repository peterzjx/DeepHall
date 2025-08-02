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


class LaughlinVelocity(nn.Module):
    """Create drift velocity for the Laughlin wavefunction."""
    nspins: tuple[int, int]
    flux: float

    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - (nelec - 1)
        assert nelec == 2 * self.Q1 + 1  # Ground state for 1/3

    def __call__(self, electrons_xy):
        Ne = sum(self.nspins)
        # zij shape: (batch, N, N, 2)
        x, y = electrons_xy[..., 0, None], electrons_xy[..., 1, None]
        r = jnp.sqrt(x**2 + y**2)
        
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
        # Zero the diagonal (j == k)
        mask = ~jnp.eye(Ne, dtype=bool)  # shape: (N, N)
        mask = mask[..., None]
        weights = weights * mask  # zero out diagonal

        # print('shapes', xij.shape, yij.shape, zij.shape, rij2.shape)
        
        # Multiply weights with direction vectors
        weighted_zij = zij * weights  # shape: (N, N, 2)
        iweighted_x_zij = x_zij * weights
        
        # Sum over j (axis 1)
        v_paired = 3.0 * jnp.sum(weighted_zij, axis=1)  # shape: (N, 2)
        iv_paired = 3.0 * jnp.sum(iweighted_x_zij, axis=1)  # shape: (N, 2)

        vx = 3 * (1 - Ne) / (1 + r**2) * x
        vy = 3 * (1 - Ne) / (1 + r**2) * y 

        # ivx = 3 * (1 - Ne) / (2 * r**2) * (-y)
        # ivy = 3 * (1 - Ne) / (2 * r**2) * x

        v = jnp.concatenate([vx, vy], axis = -1)
        # iv = jnp.concatenate([ivx, ivy], axis = -1)
        v_real = v + v_paired
        v_imag = iv_paired
        return v_real+ 1j * v_imag
