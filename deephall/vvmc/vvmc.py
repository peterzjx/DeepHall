
from typing import NamedTuple
from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
from chex import ArrayTree, PRNGKey
import deephall.vvmc.velocity_utils as v_utils
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
    # d = jnp.squeeze(d)
    # next_d = jnp.squeeze(d)
    
    dR = next_electrons_xy - electrons_xy
    _2F = jnp.real(v_xy + next_v_xy)
    element_wise = dR * _2F
    # Sum over last two dimensions
    dot_product = jnp.sum(element_wise, axis=[-1, -2])
    acceptance_threshold = jnp.exp(dot_product)
    metric = jnp.prod(d, axis = [-1, -2])
    next_metric = jnp.prod(next_d, axis = [-1, -2])
    
    acceptance_threshold = acceptance_threshold * metric / next_metric
    
    accepted_idx = jax.random.uniform(key, shape=acceptance_threshold.shape) < acceptance_threshold
    print('d, acceptance shape ', d.shape, acceptance_threshold.shape, accepted_idx.shape)
    return accepted_idx, acceptance_threshold

def calculate_move_xy(key: PRNGKey, xy: jnp.ndarray, stddev: float = 0.03):
    '''
        key: jax.random.PRNGKey
        v: velocity
        d_metric: d metric
        tau: time step
    '''
    move = (
        jax.random.normal(
            key=key,
            shape=xy.shape
        ) * stddev
    )
    move = jnp.clip(move, -100, 100)
    
    return move

# def calculate_move_thetaphi(key: PRNGKey, theta_phi: jnp.ndarray, stddev: float = 0.03):
#     # TODO: check the metrics if sin theta is needed 
#     move = (
#         jax.random.normal(
#             key=key,
#             shape=theta_phi.shape
#         ) * stddev
#     )
#     move = jnp.clip(move, -30, 30)
#     return move


def vvmc_update(key: PRNGKey, params: ArrayTree, system: System, model: LogPsiNetwork, walker_state: WalkerState, num_accepted: int):
    '''
        key: jax.random.PRNGKey
        params: network parameters
        model: psi model, callable
        walker_state: current walker state
        tau: time step
    '''
    # print_gpu_memory()
    key, key_move, key_accept = jax.random.split(key, 3)

    move_xy = calculate_move_xy(key_move, walker_state.electrons_xy, stddev=0.1)
    trial_electrons_xy = walker_state.electrons_xy +  move_xy
    
    trial_v = v_utils.batch_drift_velocity(params, model, trial_electrons_xy)
    trial_d = v_utils.calculate_d_metric_xy(trial_electrons_xy, _2Q=system.flux)

    accepted_idx, acceptance_threshold = calculate_acceptance_xy(key_accept, walker_state.electrons_xy, trial_electrons_xy, walker_state.v, trial_v, walker_state.d_metric, trial_d)
    start_step_idx = walker_state.dmc_run_step < 1 # TODO: just a small number
    accepted_idx = jnp.where(start_step_idx, jnp.ones_like(walker_state.lnpsi, dtype=bool), accepted_idx)
    # acceptance_threshold = jnp.ones_like(walker_state.lnpsi)
    num_accepted += jnp.sum(accepted_idx)

    # update the walkers according to the acceptance
    next_electrons_xy = jnp.where(accepted_idx[..., None, None], trial_electrons_xy, walker_state.electrons_xy)
    next_v = jnp.where(accepted_idx[..., None, None], trial_v, walker_state.v)
    next_d = jnp.where(accepted_idx[..., None,None], trial_d, walker_state.d_metric)

    next_local_energy = v_utils.batch_local_energy(params, system, model, next_electrons_xy)

    next_walker_state = WalkerState(
        electrons=jnp.zeros_like(walker_state.electrons),
        electrons_xy=next_electrons_xy,
        electrons_xy_move=next_electrons_xy - walker_state.electrons_xy,
        v=next_v,
        d_metric=next_d,
        lnpsi=jnp.zeros_like(walker_state.lnpsi), # dummy not updating
        local_energy=next_local_energy,
        weights=jnp.ones_like(walker_state.weights), # dummy not updating
        dmc_mean_energy=walker_state.dmc_mean_energy,
        dmc_run_step=walker_state.dmc_run_step+1
    )

    return next_walker_state, key, num_accepted, acceptance_threshold


def make_vvmc_step(system: System, network: LogPsiNetwork, batch_per_device: int, steps: int = 1):
    @jax.jit
    def vvmc_step(
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
            walker_state, key, num_accepts, acceptance_threshold = t
            return vvmc_update(key, params, system, network, walker_state, num_accepts)
        
        # TODO: fix local energy to a meaningful value
        walker_state, key, num_accepts, acceptance_threshold= lax.fori_loop(
            0, steps, step_fn, (init_walker_state, key, 0, jnp.ones_like(init_walker_state.lnpsi))  # (walker_state, key, num_accepts)
        )
        pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
        pmove = constants.pmean(pmove)
        return walker_state, pmove, acceptance_threshold
    
    return vvmc_step


def initialize_walker_state(electrons: jnp.ndarray):
    pass
    