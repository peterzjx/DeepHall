
from flax import linen as nn
from jax import numpy as jnp

from deephall.config import OrbitalType

from .blocks import Jastrow, Orbitals


class PfaffianLayers(nn.Module):
    num_heads: int
    heads_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, electrons: jnp.ndarray, spins: jnp.ndarray):
        theta, phi = electrons[..., 0], electrons[..., 1]
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

class Pfaffian(nn.Module):
    nspins: tuple[int, int]
    Q: float
    ndets: int
    num_heads: int
    heads_dim: int
    num_layers: int
    orbital_type: OrbitalType
    
    def __init__(self):
        self.net = PfaffianLayers(num_heads=1, heads_dim=1, num_layers=1)

    @nn.compact
    def __call__(self, electrons: jnp.ndarray, spins: jnp.ndarray):
        n_ij = self.net(electrons, spins)  # [N, N]
        # Antisymmetrize n_ij to create g_ij
        g_ij = (n_ij - n_ij.T) / 2  # Make it antisymmetric
        pfaffian = jnp.sqrt(jnp.linalg.det(g_ij))
        return pfaffian * self.orbitals(electrons)

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
