
from flax import linen as nn
import numpy as np
import jax
from jax import numpy as jnp
from flax.training import train_state
import optax

from deephall.config import OrbitalType, FluxType

from deephall.networks.blocks import Jastrow, Orbitals
from deephall.networks.bosonic_network import SymmetricNetwork
from deephall.networks.parton import Parton

class VelocityModel(nn.Module):
    nspins: tuple[int, int]
    Q: float
    ndets: int
    num_heads: int
    heads_dim: int
    num_layers: int
    orbital_type: OrbitalType
    flux_type: FluxType
    mask_len: float = 0.1
    benchmark_original: bool = False