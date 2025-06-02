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
    theta = electrons[..., 0]
    phi = electrons[..., 1]
    grad_fn = jax.grad(lambda x: model(params, x).real)
    batch_grad_fn = jax.vmap(grad_fn, in_axes=0)
    batch_grad_logpsi = batch_grad_fn(electrons)  # [nwalkers, nelec, 2]
    inv_JT = jnp.array([[-2 * jnp.cos(phi) * jnp.sin(theta / 2)**2, -jnp.sin(phi) * jnp.tan(theta/2)],
                           [-2 * jnp.sin(phi) * jnp.sin(theta / 2)**2,  jnp.cos(phi) * jnp.tan(theta/2)]])
    inv_JT = jnp.transpose(inv_JT, [2,3,0,1])
    drift_vxy = jnp.einsum('ijkl,ijl->ijk', inv_JT, batch_grad_logpsi)
    drift_vxy = jnp.clip(drift_vxy, -100, 100)
    return drift_vxy

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
    theta = electrons[..., 0]
    # phi = electrons[..., 1]
    r = 1.0 / (1e-10 + jnp.tan(theta / 2))    
    d_metric = (1 + r**2)**2 / (2.0 * _2Q)
    return jnp.expand_dims(d_metric, axis=-1)

def calculate_d_metric_xy(electrons_xy: jnp.ndarray, _2Q: float=9.0):
    x = electrons_xy[..., 0]
    y = electrons_xy[..., 1]
    r2 = (x**2 + y**2)    
    d_metric = (1 + r2)**2 / (2.0 * _2Q)
    return jnp.expand_dims(d_metric, axis=-1)
