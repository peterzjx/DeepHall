import flax.linen as nn
import jax.numpy as jnp
import jax
from itertools import combinations

class ModulePhase(nn.Module):
    Q: int
    nspins: int

    def setup(self):
        nelec = sum(self.nspins)
        self.Q1 = self.flux / 2 - self.cf_flux * (sum(self.nspins) - 1)
        
    @nn.compact
    def __call__(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))
        uivj = self.antisym_matrix(u, v)
        element = uivj + jnp.eye(uivj.shape[0])
        jastrow = jnp.prod(element)
        ln_laughlin = 3 / 2 * jnp.log(jastrow)
        return ln_laughlin
    
    def antisym_matrix(u, v):
        """
        u, v: [Ne]
        return: [Ne, Ne] with M_ij = u_i v_j - u_j v_i
        """
        ui, uj = u[:, None], u[None, :]
        vi, vj = v[:, None], v[None, :]
        return ui * vj - uj * vi