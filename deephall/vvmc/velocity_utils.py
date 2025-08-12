from chex import ArrayTree
from jax import numpy as jnp
from flax import linen as nn
import jax
from deephall.config import System
from deephall.types import LogPsiNetwork
from deephall import hamiltonian

def thetaphi_xy(electron_thetaphi: jnp.ndarray):
    theta = electron_thetaphi[..., 0]
    phi = electron_thetaphi[..., 1]
    x = jnp.cos(phi) / jnp.tan(theta / 2)
    y = jnp.sin(phi) / jnp.tan(theta / 2)
    electron_xy = jnp.stack([x, y], axis=-1)
    return electron_xy
def xy_thetaphi(electron_xy: jnp.ndarray):
    x = electron_xy[..., 0]
    y = electron_xy[..., 1]
    r = jnp.sqrt(x**2+y**2)
    phi = jnp.arctan2(y, x)
    theta = 2.0 * jnp.arctan(1e-10+1.0 / r)
    electron_thetaphi = jnp.stack([theta, phi], axis=-1)
    return electron_thetaphi

def batch_drift_velocity(params: ArrayTree, model: LogPsiNetwork, electrons_xy: jnp.ndarray):
    """
        electrons: [nwalkers, nelec, 2]
    """
    drift_F = lambda x: model(params, x)
    batch_drift_fn = jax.vmap(drift_F, in_axes=0)
    drift_vxy = batch_drift_fn(electrons_xy)  # [nwalkers, nelec, 2]
    return drift_vxy

def batch_local_energy(params: ArrayTree, system: System, v_model: LogPsiNetwork, electrons: jnp.ndarray):
    """
        electrons: [nwalkers, nelec, 2]
    """
    # hamiltonian.local_energy takes non-batched electrons
    local_energy_fn = hamiltonian.local_v_energy(v_model, system)
    batch_local_energy = jax.vmap(local_energy_fn, in_axes=(None, 0))
    return batch_local_energy(params, electrons)[0].real  # only take total energy

def calculate_d_metric_xy(electrons_xy: jnp.ndarray, _2Q: float):
    x = electrons_xy[..., 0]
    y = electrons_xy[..., 1]
    r2 = (x**2 + y**2)    
    d_metric = (1 + r2)**2 / (2.0 * _2Q)
    return jnp.expand_dims(d_metric, axis=-1)
