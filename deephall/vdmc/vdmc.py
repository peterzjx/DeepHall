
from typing import NamedTuple
from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
from chex import ArrayTree, PRNGKey
import deephall.dmc.velocity_utils as v_utils
from deephall import constants
from deephall.types import WalkerState, LogPsiNetwork
from deephall.config import Config, System
########################################################################################
# import os
# import subprocess

# def print_gpu_memory():
#     result = subprocess.run(
#         ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
#         stdout=subprocess.PIPE,
#         text=True
#     )
#     print(f"[MEM] GPU Memory Used: {result.stdout.strip()} MB")
# ########################################################################################

_Z_MAX = 1e9
_Z_MIN = 1e-9

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


def wrap_coord(coord):
    pi = jnp.pi
    two_pi = 2 * jnp.pi

    theta = coord[..., 0]
    phi = coord[..., 1]

    # theta > pi: reflect theta, shift phi
    over = theta > pi
    theta = jnp.where(over, 2 * pi - theta, theta)
    phi = jnp.where(over, phi + pi, phi)

    # theta < 0: reflect theta, shift phi
    under = theta < 0
    theta = jnp.where(under, -theta, theta)
    phi = jnp.where(under, phi - pi, phi)

    # wrap phi to [0, 2π]
    phi = phi % two_pi

    # reassemble coord
    coord = coord.at[..., 0].set(theta)
    coord = coord.at[..., 1].set(phi)

    return coord


# def log_green_function_branching(local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
#     '''
#         local_energy: current local energy
#         next_local_energy: next local energy
#     '''
#     # print('energy shape', next_local_energy.shape)
#     # print('kappa_tau', kappa_tau)
#     # print('local_energy', local_energy.shape)
#     # print('next_local_energy', next_local_energy.shape)
#     # print('total_mean_energy', total_mean_energy)
#     return -kappa_tau * (next_local_energy + local_energy - 2 * total_mean_energy) / 2


# def reweight_walkers(weights: jnp.ndarray, local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
#     weights = weights * jnp.exp(log_green_function_branching(local_energy, next_local_energy, kappa_tau, total_mean_energy))
#     n_walkers = weights.shape[0]
#     weights = jnp.sqrt(n_walkers) * weights / jnp.linalg.norm(weights)  # TODO: check if this is correct    
#     return weights


def calculate_acceptance(key: PRNGKey, electrons: jnp.ndarray, next_electrons: jnp.ndarray, lnpsi: jnp.ndarray, next_lnpsi: jnp.ndarray, v: jnp.ndarray, next_v: jnp.ndarray, d: float, next_d: float, tau: float):
    '''
        key: jax.random.PRNGKey
        electrons: electrons coordinates
        next_electrons: next electrons coordinates
        psi: current psi
        next_psi: next psi
        v: current velocity
        next_v: next velocity
        d: d metric
        next_d: next d metric
        tau: time step
    '''
    electrons_xy = thetaphi_xy(electrons)
    next_electrons_xy = thetaphi_xy(next_electrons)
    next_electrons_norm = jnp.abs(next_electrons_xy[..., 0]) + jnp.abs(next_electrons_xy[..., 1])
    log_green_function_forward = log_green_function(electrons_xy, next_electrons_xy, v, d, tau)
    log_green_function_backward = log_green_function(next_electrons_xy, electrons_xy, next_v, next_d, tau)

    # acceptance_threshold = jnp.exp(2.0 * (jnp.real(next_lnpsi) - jnp.real(lnpsi)))  * jnp.exp(log_green_function_backward - log_green_function_forward)
    # acceptance_threshold = jnp.exp(2.0 * (jnp.abs(next_lnpsi) - jnp.abs(lnpsi)))
    walkers_size = acceptance_threshold.shape[0]
    accepted_idx = jax.random.uniform(key, shape=(walkers_size,)) < acceptance_threshold and jnp.linalg.norm(electrons_xy,axis=tuple(range(1, electrons_xy.ndim)))<_Z_MAX*jax.ones(shape=(walkers_size,)) and jnp.linalg.norm(electrons_xy,axis=tuple(range(1, electrons_xy.ndim)))>_Z_MIN*jax.ones(shape=(walkers_size,))
    return accepted_idx, acceptance_threshold, log_green_function_forward, log_green_function_backward

def calculate_acceptance_xy(key: PRNGKey, electrons_xy: jnp.ndarray, next_electrons_xy: jnp.ndarray, v_xy: jnp.ndarray, next_v_xy: jnp.ndarray, d: float, next_d: float):
    '''
        key: jax.random.PRNGKey
        electrons: electrons coordinates
        next_electrons: next electrons coordinates
        psi: current psi
        next_psi: next psi
        v: current velocity
        next_v: next velocity
        d: d metric
        next_d: next d metric
        tau: time step
    '''
    dR = next_electrons_xy - electrons_xy
    _2F = jnp.real(v_xy + next_v_xy)
    
    acceptance_threshold = jnp.exp(jnp.vdot(dR, _2F))
    theta = xy_thetaphi(electrons_xy)[..., 0]
    next_theta = xy_thetaphi(next_electrons_xy)[..., 0]
    metric = jnp.prod(jnp.sin(theta), axis = -1)
    next_metric = jnp.prod(jnp.sin(next_theta), axis = -1)
    # acceptance_threshold = acceptance_threshold * next_metric / metric
    acceptance_threshold = 1.0 * next_metric / metric
    accepted_idx = jax.random.uniform(key, shape=acceptance_threshold.shape) < acceptance_threshold
    return accepted_idx, acceptance_threshold

def calculate_move_xy(key: PRNGKey, v: jnp.ndarray, d_metric: float, tau: float = 0.02):
    '''
        key: jax.random.PRNGKey
        v: velocity
        d_metric: d metric
        tau: time step
    '''
    move = (
        jax.random.normal(
            key=key,
            shape=v.shape
        ) * jnp.sqrt(d_metric * tau) 
    )
    move = jnp.clip(move, -100, 100)
    
    return move

def calculate_move_thetaphi(key: PRNGKey, theta_phi: jnp.ndarray, stddev: float = 0.03):
    # TODO: check the metrics if sin theta is needed 
    move = (
        jax.random.normal(
            key=key,
            shape=theta_phi.shape
        ) * stddev
    )
    move = jnp.clip(move, -30, 30)
    return move


def vdmc_update(key: PRNGKey, params: ArrayTree, system: System, model: LogPsiNetwork, walker_state: WalkerState, num_accepted: int):
    '''
        key: jax.random.PRNGKey
        params: network parameters
        model: psi model, callable
        walker_state: current walker state
        tau: time step
    '''
    # print_gpu_memory()
    key, key_move, key_accept = jax.random.split(key, 3)

    # theta = walker_state.electrons[..., 0]
    # phi = walker_state.electrons[..., 1]
    
    
    # xy_move = calculate_move_xy(key_move, walker_state.v, walker_state.d_metric, tau=0.02)
    # trial_electrons_xy = walker_state.electrons_xy + xy_move
    # trial_electrons = xy_thetaphi(trial_electrons_xy)
    # trial_electrons = wrap_coord(trial_electrons) #Adjusting points that are too close to poles
    # trial_electrons_xy = thetaphi_xy(trial_electrons)

    move_thetaphi = calculate_move_thetaphi(key_move, walker_state.electrons, stddev=0.01)
    trial_electrons = walker_state.electrons +  move_thetaphi
    trial_electrons = wrap_coord(trial_electrons) #Adjusting points that are too close to poles
    trial_electrons_xy = thetaphi_xy(trial_electrons)
    
    next_v = v_utils.batch_drift_velocity(params, model, trial_electrons)
    next_d = v_utils.calculate_d_metric_xy(trial_electrons_xy, _2Q=system.flux)

    accepted_idx, acceptance_threshold = calculate_acceptance_xy(key_accept, walker_state.electrons_xy, trial_electrons_xy, walker_state.v, next_v, walker_state.d_metric, next_d)
    start_step_idx = walker_state.dmc_run_step < 1 # TODO: just a small number
    accepted_idx = jnp.where(start_step_idx, jnp.ones_like(walker_state.lnpsi, dtype=bool), accepted_idx)
    # acceptance_threshold = jnp.ones_like(walker_state.lnpsi)
    num_accepted += jnp.sum(accepted_idx)

    # update the walkers according to the acceptance
    next_electrons_xy = jnp.where(accepted_idx[..., None, None], trial_electrons_xy, walker_state.electrons_xy)
    next_electrons = jnp.where(accepted_idx[..., None, None], trial_electrons, walker_state.electrons)
    next_v = jnp.where(accepted_idx[..., None, None], next_v, walker_state.v)
    # next_lnpsi = jnp.where(accepted_idx, next_lnpsi, walker_state.lnpsi)
    next_d = jnp.where(accepted_idx[..., None,None], next_d, walker_state.d_metric)

    next_local_energy = v_utils.batch_local_energy(params, system, model, next_electrons)
    # move = jnp.where(accepted_idx[..., None, None], move, jnp.zeros_like(move))
    # move = trial_electrons - walker_state.electrons
    # xy_move = jnp.where(accepted_idx[..., None, None], xy_move, jnp.zeros_like(xy_move))

    next_walker_state = WalkerState(
        electrons=next_electrons,
        electrons_xy=next_electrons_xy,
        v=next_v,
        d_metric=next_d,
        lnpsi=jnp.zeros_like(walker_state.lnpsi), # dummy not updating
        local_energy=next_local_energy,
        weights=jnp.ones_like(walker_state.weights), # dummy not updating
        dmc_mean_energy=walker_state.dmc_mean_energy,
        dmc_run_step=walker_state.dmc_run_step+1
    )

    return next_walker_state, key, num_accepted


def make_vdmc_step(system: System, network: LogPsiNetwork, batch_per_device: int, steps: int = 10):
    @jax.jit
    def vdmc_step(
        params: ArrayTree, init_walker_state: WalkerState, key: PRNGKey,
    ):
        """Performs a set of DMC steps.

        Args:
        params: parameters to pass to the batch_network.
        data: (batched) DMC configurations to pass to the batch_network.
        key: RNG state.

        Returns:
        (data, pmove), where data is the updated DMC configurations, key the
        updated RNG state and pmove the average probability a move was accepted.
        """
        
        def step_fn(i, t):
            walker_state, key, num_accepts = t
            return vdmc_update(key, params, system, network, walker_state, num_accepts)
        
        # TODO: fix local energy to a meaningful value
        walker_state, key, num_accepts= lax.fori_loop(
            0, steps, step_fn, (init_walker_state, key, 0)  # (walker_state, key, num_accepts)
        )
        pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
        pmove = constants.pmean(pmove)
        return walker_state, pmove
    
    return vdmc_step


def initialize_walker_state(electrons: jnp.ndarray):
    pass
    