
from flax import linen as nn
import jax
from jax import numpy as jnp
from flax.training import train_state
import optax

from deephall.config import OrbitalType

from .blocks import Jastrow, Orbitals

class PairwiseNetwork(nn.Module):
    """Feedforward NN that maps (ri, rj) to a complex number."""
    hidden_dim: int = 64  # Adjustable hidden layer size

    @nn.compact
    def __call__(self, rij):
        """Forward pass: Input shape [batch, Ne, Ne, 4] -> Output [batch, Ne, Ne, 2]"""
        x = nn.Dense(self.hidden_dim)(rij)
        x = nn.softplus(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(2)(x)  # Output two values (real & imag)
        return x[..., 0] + 1j * x[..., 1]  # Convert to complex

def extract_pairs(electron):
    """Generate unordered (ri, rj) pairs from electron coordinates."""
    theta, phi = electron[..., 0], electron[..., 1]
    u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [..., N, 1]
    v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [..., N, 1]
    Ne, _ = electron.shape
    # idx_i, idx_j = jnp.triu_indices(Ne)  # Get indices for i ≤ j to enforce ordering
    idx_i, idx_j = jnp.meshgrid(jnp.arange(Ne), jnp.arange(Ne), indexing='ij')
    # Gather the corresponding (ri, rj) pairs
    # ri = electron[idx_i, :]  # Shape: [Ne, 2]
    # rj = electron[idx_j, :]  # Shape: [Ne, 2]
    ui = u[idx_i]  # Shape: [Ne, 2]
    vi = v[idx_i]  # Shape: [Ne, 2]
    uj = u[idx_j]  # Shape: [Ne, 2]
    vj = v[idx_j]  # Shape: [Ne, 2]

    rij = jnp.concatenate([ui, vi, uj, vj], axis=-1)  # Shape: [batch, num_pairs, 4]
    return rij    

def original_pfaf(electron):
    """Generate unordered (ri, rj) pairs from electron coordinates."""
    theta, phi = electron[..., 0], electron[..., 1]
    u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [..., N, 1]
    v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [..., N, 1]
    Ne, _ = electron.shape
    # # idx_i, idx_j = jnp.triu_indices(Ne)  # Get indices for i ≤ j to enforce ordering
    # idx_i, idx_j = jnp.meshgrid(jnp.arange(Ne), jnp.arange(Ne), indexing='ij')
    # # Gather the corresponding (ri, rj) pairs
    # ui = u[idx_i]  # Shape: [Ne, 2]
    # vi = v[idx_i]  # Shape: [Ne, 2]
    # uj = u[idx_j]  # Shape: [Ne, 2]
    # vj = v[idx_j]  # Shape: [Ne, 2]

    # pf_ij = jnp.concatenate(1.0 / (ui*vj - uj*vi), axis=-1)  # Shape: [batch, num_pairs, 4]
    # # pf_ij = jnp.nan_to_num(pf_ij, nan=0, posinf=0, neginf=0)
    # return pf_ij    
    pf_ij = ( 1 - jnp.eye(Ne)) / (u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0]) + 1e-10)  # [..., N, N]
    return pf_ij

class Pfaffian(nn.Module):
    nspins: tuple[int, int]
    Q: float
    ndets: int
    num_heads: int
    heads_dim: int
    num_layers: int
    orbital_type: OrbitalType

    @nn.compact
    def __call__(self, electrons: jnp.ndarray):
        theta, phi = electrons[..., 0], electrons[..., 1]
        spins = jnp.array([1] * self.nspins[0] + [-1] * self.nspins[1])
        
        r_ij = extract_pairs(electron=electrons)
        model = PairwiseNetwork()
        params = model.init(jax.random.PRNGKey(42), r_ij)  # Initialize model
        n_ij = model.apply(params, r_ij)  # Forward pass
        g_ij = (n_ij - n_ij.T) / 2  # Make it antisymmetric
        # g_ij = original_pfaf(electron=electrons)
        pfaffian = jnp.sqrt(jnp.linalg.det(g_ij))
        # print(pfaffian * self.flux_attachment(electrons))
        return jnp.log(self.flux_attachment(electrons) * pfaffian)

    @nn.compact
    def flux_attachment(self, electrons):
        """
            electrons: [..., N, 2]
        """
        theta, phi = electrons[..., 0], electrons[..., 1]
        
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [..., N, 1]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [..., N, 1]

        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])  # [..., N, N]
        # uivj - ujvi + δij == (zi-zj) for i!=j, 1 for i==j
        # Pfaffian wavefunction : Pf[G(vi, ui)] * Prod_{i<j}(zi-zj)**2
        #                       = Pf[G(vi, ui)] * Prod_{i,j}(element_ij)

        orbitals = jnp.prod(element, keepdims=False)  # [..., N, 1]   
        # Prod_j (zi-zj), i!=j
    
        return orbitals
