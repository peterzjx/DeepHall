from chex import ArrayTree
from jax import numpy as jnp
from flax import linen as nn
import jax
from deephall.config import System
from deephall.types import LogPsiNetwork
from deephall import hamiltonian

def batch_drift_velocity(params: ArrayTree, model: LogPsiNetwork, electrons: jnp.ndarray):
    """
        electrons: [nwalkers, nelec, 2]
    """
    # TODO: convert between xy and theta, phi

    # def helper(params, electrons):
    #     print('electrons in helper', electrons.shape)
    #     print('after unsqueeze', electrons[None, ...].shape)
    #     model_output = model(params, electrons[None, ...]).real
    #     return jnp.squeeze(model_output, axis=0)
    print('electrons in drift velocity', electrons.shape)
    grad_fn = jax.grad(lambda x: model(params, x).real)
    batch_grad_fn = jax.vmap(grad_fn, in_axes=0)
    batch_grad_logpsi = batch_grad_fn(electrons)  # [nwalkers, nelec, 2]
    print('batch_grad_logpsi', batch_grad_logpsi)
    return batch_grad_logpsi

def batch_log_psi(params: ArrayTree, model: LogPsiNetwork, electrons: jnp.ndarray):
    batch_model = jax.vmap(model, in_axes=(None, 0))
    logpsi = batch_model(params, electrons).real
    return logpsi

def batch_local_energy(params: ArrayTree, system: System, model: LogPsiNetwork, electrons: jnp.ndarray):
    # hamiltonian.local_energy takes non-batched electrons
    local_energy_fn = hamiltonian.local_energy(model, system)
    batch_local_energy = jax.vmap(local_energy_fn, in_axes=(None, 0))
    return batch_local_energy(params, electrons)[0].real  # only take total energy

def calculate_d_metric(electrons: jnp.ndarray, _2Q: float=9.0):
    # TODO: change input to theta and phi
    x = electrons[..., 0]
    y = electrons[..., 1]
    d_metric = (1 + x**2 + y**2)**2 / (2.0 * _2Q)
    return jnp.expand_dims(d_metric, axis=-1)
