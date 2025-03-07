
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
        """Forward pass: Input shape [batch, N_pairs, D] -> Output [batch, N_pairs, 2]"""
        x = nn.Dense(self.hidden_dim, use_bias=True)(rij)
        x = nn.sigmoid(x)  # Smooth activation
        
        # Second hidden layer
        x = nn.Dense(self.hidden_dim, use_bias=True)(x)
        x = nn.sigmoid(x)
        
        # Third hidden layer
        x = nn.Dense(self.hidden_dim, use_bias=True)(x)
        x = nn.sigmoid(x)

        # Output layer
        x = nn.Dense(2, use_bias=True)(x)
        return x[..., 0] + 1j * x[..., 1]

def extract_pairs(electron):
    """
    Generate unordered (ri, rj) pairs from electron coordinates.
    Returns: [Ne, Ne, D]
    D = length(features: ui, vi, ...)
    """
    theta, phi = electron[..., 0], electron[..., 1]
    # u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [N, 1]
    # v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [N, 1]
    Ne, _ = electron.shape
    idx_i, idx_j = jnp.meshgrid(jnp.arange(Ne), jnp.arange(Ne), indexing='ij')

    # Gather the corresponding (ri, rj) pairs
    thetai = jnp.reshape(theta[idx_i], [-1])  # Shape: [Ne * Ne]
    phii = jnp.reshape(phi[idx_i], [-1])  # Shape: [Ne * Ne]
    thetaj = jnp.reshape(theta[idx_j], [-1])  # Shape: [Ne * Ne]
    phij = jnp.reshape(phi[idx_j], [-1])  # Shape: [Ne * Ne]

    rij = jnp.stack([thetai, phii, thetaj, phij], axis=-1)  # [Ne*Ne, D]
    print("rij", rij.shape)
    
    return rij    

def original_pfaf(electron):
    """Generate unordered (ri, rj) pairs from electron coordinates."""
    theta, phi = electron[..., 0], electron[..., 1]
    u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]  # [..., N, 1]
    v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]  # [..., N, 1]
    Ne, _ = electron.shape   
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
        # Using NN for wfn
        Ne = electrons.shape[0]
        pair_feature = extract_pairs(electron=electrons)
        model = PairwiseNetwork()
        params = model.init(jax.random.PRNGKey(42), pair_feature)  # Initialize model
        model_output = model.apply(params, pair_feature)  # Forward pass
        n_ij = jnp.reshape(model_output, [Ne, Ne])
        g_ij = (n_ij - n_ij.T) / 2  # Make it antisymmetric
        # Using original Moore-Read Pfaffian for benchmarking
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
