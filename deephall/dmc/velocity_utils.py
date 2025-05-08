from chex import ArrayTree
from jax import numpy as jnp
from flax import linen as nn
import jax

def drift_velocity(params: ArrayTree, model: nn.Module, electrons: jnp.ndarray):
    psi = psi(params, model, electrons)
    grad_psi = jax.grad(psi, argnums=1)
    return grad_psi(params, model, electrons)

def psi(params: ArrayTree, model: nn.Module, electrons: jnp.ndarray):
    # TODO: take modulus
    return model.apply({'params': params}, electrons)

def local_energy(params: ArrayTree, model: nn.Module, electrons: jnp.ndarray):
    # TODO: import or copy from hamiltonian.py
    pass

def calculate_d_metric(electrons: jnp.ndarray, _2Q: float=9.0):
    # TODO: change input to theta and phi
    x = electrons[..., 0]
    y = electrons[..., 1]
    tmp = 1 + x**2 + y**2
    return tmp**2 / (2.0 * _2Q)