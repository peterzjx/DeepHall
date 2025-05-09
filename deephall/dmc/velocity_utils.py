from chex import ArrayTree
from jax import numpy as jnp
from flax import linen as nn
import jax
from deephall.types import LogPsiNetwork
from deephall import hamiltonian

def drift_velocity(params: ArrayTree, model: LogPsiNetwork, electrons: jnp.ndarray):
    """
        electrons: [nwalkers, nelec, 2]
    """
    # TODO: convert between xy and theta, phi

    def helper(params, electrons):
        print('electrons in helper', electrons[None, ...].shape)
        model_output = model(params, electrons[None, ...]).real
        return jnp.squeeze(model_output, axis=0)

    grad_fn = jax.grad(lambda x: helper(params, x))
    # Vectorize over the walker dimension
    batched_grad = jax.vmap(grad_fn, in_axes=0)
    print('electrons', electrons.shape)
    grad_logpsi = batched_grad(electrons)  # [nwalkers, nelec, 2]
    print('grad_logpsi', grad_logpsi)
    return grad_logpsi

def log_psi(params: ArrayTree, model: LogPsiNetwork, electrons: jnp.ndarray):
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
    d_metric = (1 + x**2 + y**2)**2 / (2.0 * _2Q)
    return jnp.expand_dims(d_metric, axis=-1)
