import flax.linen as nn
import jax.numpy as jnp
import jax
from itertools import combinations
# from .bosonic_network import SymmetricAttNetwork, SymmetricProductAttNetwork, SymmetricProductMLPNetwork, SymmetricMLPNetwork

class SymmetricMLPNetwork(nn.Module):
    @nn.compact
    def __call__(self, electrons):
        '''
            electrons: [..., N, 2], theta and phi, both real numbers
            returns: [..., 2], a, b, where a+bi is the value of the wavefunction
        '''
        theta, phi = electrons[..., 0], electrons[..., 1]  # [..., N], [..., N]
        uv = jnp.stack([jnp.cos(theta), jnp.cos(phi / 2)], axis= -1)
        # three MLP layers
        feature = nn.Dense(32)(uv)  # [..., N, 64]
        feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature = nn.sigmoid(feature)
        feature = nn.Dense(128)(feature)  # [..., N, 128]
        feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature = nn.sigmoid(feature)
        feature = nn.Dense(32)(feature)  # [..., N, 64]
        feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature = nn.sigmoid(feature)
        feature = jnp.max(feature, axis=-2)  # [..., 64]
        flux = nn.Dense(1)(feature)  # [..., 2]
        
        return jnp.log(flux ** 2)

def antisym_matrix(u, v):
    """
    u, v: [Ne]
    return: [Ne, Ne] with M_ij = u_i v_j - u_j v_i
    """
    ui, uj = u[:, None], u[None, :]
    vi, vj = v[:, None], v[None, :]
    return ui * vj - uj * vi

# def pairwise_diffs(z: jnp.ndarray) -> jnp.ndarray:
#     """
#     Compute zi - zj for all i,j for z shaped (Ne, 2).

#     Returns:
#         diffs: array shape (Ne, Ne, 2) with diffs[i, j] == z[i] - z[j].
#     """
#     return z[:, None, :] - z[None, :, :]
def pairwise_unique_diffs(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute zi - zj only for unique pairs i < j.

    Returns:
        diffs: array shape (M, 2) where M = Ne*(Ne-1)//2
        idxs: array shape (M, 2) with corresponding (i,j) indices
    """
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("z must have shape (Ne, 2)")
    Ne = z.shape[0]
    if Ne < 2:
        return jnp.empty((0, 2)), jnp.empty((0, 2), dtype=int)
    iu = jnp.triu_indices(Ne, k=1)
    diffs_full = z[:, None, :] - z[None, :, :]
    diffs = diffs_full[iu]            # shape (M, 2)
    idxs = jnp.vstack(iu).T           # shape (M, 2)
    return diffs, idxs

class RelativePhaseNetwork(nn.Module):
    dim: int = 16
    layers: int = 2
    @nn.compact
    def __call__(self, zij):
        xij, yij = zij[..., 0], zij[..., 1]
        arg = jnp.arctan2(yij, xij)
        return 3 / 2 * arg
        x = zij
        for _ in range(self.layers):
            x = nn.Dense(self.dim)(x)
            x = nn.sigmoid(x)
        x = nn.Dense(1)(x)
        return x

class PhaseNetwork(nn.Module):
    Q: int
    nspins: int

    def setup(self):
        self.nelec = self.nspins
        self.relative_phase_net = RelativePhaseNetwork()
        
    @nn.compact
    def __call__(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))
        z = u / v
        z = jnp.stack([z.real, z.imag], axis=-1)
        zij, zij_idxs = pairwise_unique_diffs(z)
        phase = self.relative_phase_net(zij)
        phase = jnp.sum(phase)
        return phase

    
class ModulusPhase(nn.Module):
    Q: int
    nspins: int

    def setup(self):
        self.nelec = self.nspins
        self.phase_network = PhaseNetwork(Q=self.Q, nspins=self.nspins)
        self.modulus_network = SymmetricMLPNetwork()
    @nn.compact
    def __call__(self, electrons):
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))
        uivj = antisym_matrix(u, v)
        element = uivj + jnp.eye(uivj.shape[0])
        jastrow = jnp.prod(element)
        ln_laughlin = 3 / 2 * jnp.log(jastrow)
        # modulus = jnp.real(ln_laughlin)
        phase = self.phase_network(electrons)
        # phase = ln_laughlin.imag
        # modulus = self.modulus_network(electrons)
        
        modulus = ln_laughlin.real
        phase = ln_laughlin.imag
        return modulus + 1j * phase
        
    

