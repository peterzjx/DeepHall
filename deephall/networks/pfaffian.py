
from flax import linen as nn
import jax
from jax import numpy as jnp
from flax.training import train_state
import optax

from deephall.config import OrbitalType

from .blocks import Jastrow, Orbitals

class SelfAttentionNetwork(nn.Module):
    hidden_dim: int  # Number of hidden units

    @nn.compact
    def __call__(self, hi, hj, context):
        x = jnp.concatenate([hi, hj, context], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)  # Output a single value
        return x
class PfaffianLayers(nn.Module):
    num_heads: int
    heads_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, electrons: jnp.ndarray, spins: jnp.ndarray):
        theta, phi = electrons[..., 0], electrons[..., 1]
        # u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        # v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]
        h_one = self.input_feature(theta, phi, spins)
        attention_dim = self.num_heads * self.heads_dim
        h_one = nn.Dense(attention_dim, use_bias=False)(h_one)
        for _ in range(self.num_layers):
            attn_out = nn.MultiHeadAttention(num_heads=self.num_heads)(h_one)
            h_one += nn.Dense(attention_dim, use_bias=False)(attn_out)
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
            h_one += nn.tanh(nn.Dense(attention_dim)(h_one))
            h_one = nn.LayerNorm(epsilon=1e-5)(h_one)
        return h_one

    def input_feature(self, theta: jnp.ndarray, phi: jnp.ndarray, spins: jnp.ndarray):
        return jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                spins,
            ],
            axis=-1,
        )
def compute_n_matrix(electrons, model, params):
    """Computes the n_ij matrix using a self-attention network."""
    theta, phi = electrons[..., 0], electrons[..., 1]
    u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
    v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]

    # Convert spinors to real-valued features
    h = jnp.concatenate([u.real, u.imag, v.real, v.imag], axis=-1)  # Shape: (N, 4)

    N = h.shape[0]
    n_matrix = jnp.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # Avoid self-interaction

            # Compute context as mean over other electrons
            context = jnp.mean(jnp.delete(h, [i, j], axis=0), axis=0) if N > 2 else jnp.zeros_like(h[0])

            # Compute attention score
            n_matrix = n_matrix.at[i, j].set(model.apply(params, h[i], h[j], context))

    return n_matrix

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
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]
        h = jnp.concatenate([u.real, u.imag, v.real, v.imag], axis=-1)  # Shape: (N, 4)
        n_ij = PfaffianLayers(
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            heads_dim=self.heads_dim,
        )(electrons, spins)
        # Antisymmetrize n_ij to create g_ij
        g_ij = (n_ij - n_ij.T) / 2  # Make it antisymmetric
        pfaffian = jnp.sqrt(jnp.linalg.det(g_ij))
        return pfaffian
        # return pfaffian * self.orbitals(electrons)

    @nn.compact
    def orbitals(self, electrons):
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

        orbitals = jnp.prod(element, axis=-1, keepdims=True)  # [..., N, 1]   
        # Prod_j (zi-zj), i!=j
    
        return orbitals
