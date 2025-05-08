from chex import ArrayTree
from jax import numpy as jnp
from flax import linen as nn
import jax
from deephall.types import LogPsiNetwork
from deephall import hamiltonian

def drift_velocity(params: ArrayTree, model: LogPsiNetwork, electrons: jnp.ndarray):
    grad_logpsi = jax.jacobian(model, argnums=1)
    # print('grad_lnpsi', grad_logpsi)
    return grad_logpsi(params, electrons)

def logPsi(params: ArrayTree, model: LogPsiNetwork, electrons: jnp.ndarray):
    # TODO: take modulus
    logpsi = model(params, electrons).real
    return logpsi

def local_energy(params: ArrayTree, model: LogPsiNetwork, electrons: jnp.ndarray):
    pass
    # TODO: import or copy from hamiltonian.py
    # energy = hamiltonian.local_energy(model, system: System)(electrons)
    # return energy

def calculate_d_metric(electrons: jnp.ndarray, _2Q: float=9.0):
    # TODO: change input to theta and phi
    x = electrons[..., 0]
    y = electrons[..., 1]
    tmp = 1 + x**2 + y**2
    tmp = tmp**2 / (2.0 * _2Q)
    return tmp