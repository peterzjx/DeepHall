
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


def log_green_function_branching(local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
    '''
        local_energy: current local energy
        next_local_energy: next local energy
    '''
    # print('energy shape', next_local_energy.shape)
    # print('kappa_tau', kappa_tau)
    # print('local_energy', local_energy.shape)
    # print('next_local_energy', next_local_energy.shape)
    # print('total_mean_energy', total_mean_energy)
    return -kappa_tau * (next_local_energy + local_energy - 2 * total_mean_energy) / 2


def reweight_walkers(weights: jnp.ndarray, local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
    weights = weights * jnp.exp(log_green_function_branching(local_energy, next_local_energy, kappa_tau, total_mean_energy))
    n_walkers = weights.shape[0]
    weights = jnp.sqrt(n_walkers) * weights / jnp.linalg.norm(weights)  # TODO: check if this is correct    
    return weights


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

    acceptance_threshold = jnp.exp(2.0 * (jnp.real(next_lnpsi) - jnp.real(lnpsi)))  * jnp.exp(log_green_function_backward - log_green_function_forward)
    # acceptance_threshold = jnp.exp(2.0 * (jnp.abs(next_lnpsi) - jnp.abs(lnpsi)))
    walkers_size = acceptance_threshold.shape[0]
    accepted_idx = jax.random.uniform(key, shape=(walkers_size,)) < acceptance_threshold and jnp.linalg.norm(electrons_xy,axis=tuple(range(1, electrons_xy.ndim)))<_Z_MAX*jax.ones(shape=(walkers_size,)) and jnp.linalg.norm(electrons_xy,axis=tuple(range(1, electrons_xy.ndim)))>_Z_MIN*jax.ones(shape=(walkers_size,))
    return accepted_idx, acceptance_threshold, log_green_function_forward, log_green_function_backward

def calculate_acceptance_xy(key: PRNGKey, electrons_xy: jnp.ndarray, next_electrons_xy: jnp.ndarray, lnpsi: jnp.ndarray, next_lnpsi: jnp.ndarray, v: jnp.ndarray, next_v: jnp.ndarray, d: float, next_d: float, tau: float):
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
    log_green_function_forward = log_green_function(electrons_xy, next_electrons_xy, v, d, tau)
    log_green_function_backward = log_green_function(next_electrons_xy, electrons_xy, next_v, next_d, tau)

    acceptance_threshold = jnp.exp(2.0 * (jnp.real(next_lnpsi) - jnp.real(lnpsi)))  * jnp.exp(log_green_function_backward - log_green_function_forward)
    # acceptance_threshold = jnp.exp(2.0 * (jnp.real(next_lnpsi) - jnp.real(lnpsi)))
    # acceptance_threshold = jnp.abs(jnp.exp(2.0 * ((next_lnpsi) - (lnpsi))))
    # acceptance_threshold = jnp.exp(log_green_function_backward - log_green_function_forward)
    # walkers_size = acceptance_threshold.shape[0]
    # accepted_idx = jax.random.uniform(key, shape=(walkers_size,)) < acceptance_threshold
    accepted_idx = jax.random.uniform(key, shape=log_green_function_backward.shape) < acceptance_threshold
    return accepted_idx, acceptance_threshold, log_green_function_forward, log_green_function_backward

def calculate_move(key: PRNGKey, v: jnp.ndarray, d_metric: float, tau: float):
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
        + v * tau * d_metric
    )
    move = jnp.clip(move, -100, 100)
    
    return move

def calculate_move_thetaphi(key: PRNGKey, xy: jnp.ndarray):
    theta_phi = xy_thetaphi(xy)
    move = (
        jax.random.normal(
            key=key,
            shape=theta_phi.shape
        ) * 0.3
    )
    theta_phi += move
    xy_prime = thetaphi_xy(theta_phi)
    return xy_prime - xy

def log_green_function(electrons_from: jnp.ndarray, electrons_to: jnp.ndarray, v_from: jnp.ndarray, d: float, tau: float):
    '''
        electrons_from: electrons coordinates
        electrons_to: electrons coordinates
        v_from: velocity corresponding to electrons_from
        d: d metric
        tau: time step
    '''
    displacement = electrons_to - electrons_from - v_from * d * tau  # (n_walkers, n_electrons, 2)
    squared_distances = jnp.sum(displacement ** 2, axis=-1)  # (n_walkers, n_electrons)

    expo = -0.5 * squared_distances / (jnp.squeeze(d, axis=-1) * tau)  # (n_walkers, n_electrons)

    # print('expo', expo.shape)

    # Sum over electrons and add log term
    return jnp.sum(expo, axis=1) - 2.0 * jnp.sum(jnp.log(jnp.squeeze(d, axis=-1)), axis=1)


def dmc_update(key: PRNGKey, params: ArrayTree, system: System, model: LogPsiNetwork, walker_state: WalkerState, num_accepted: int, tau: float):
    '''
        key: jax.random.PRNGKey
        params: network parameters
        model: psi model, callable
        walker_state: current walker state
        tau: time step
    '''
    key, key_move, key_accept = jax.random.split(key, 3)

    theta = walker_state.electrons[..., 0]
    phi = walker_state.electrons[..., 1]
    
    
    xy_move = calculate_move(key_move, walker_state.v, walker_state.d_metric, tau)
    # xy_move = calculate_move_thetaphi(key=key_move, xy=walker_state.electrons_xy)
    # inv_J = jnp.array([[-2 * jnp.cos(phi) * jnp.sin(theta / 2)**2, -2 * jnp.sin(phi) * jnp.sin(theta / 2)**2],
    #                        [-jnp.sin(phi) * jnp.tan(theta/2),  jnp.cos(phi) * jnp.tan(theta/2)]])
    # inv_J = jnp.transpose(inv_J, [2,3,0,1])
    # move = jnp.einsum('ijkl,ijl->ijk', inv_J, xy_move)

    
    
    # trial_electrons = walker_state.electrons + move #TODO: make thete within [0, Pi] and phi [0,2pi]
    # trial_electrons = wrap_coord(trial_electrons)
    # ele_xy = thetaphi_xy(walker_state.electrons)
    trial_electrons_xy = walker_state.electrons_xy + xy_move
    trial_electrons = xy_thetaphi(trial_electrons_xy)
    trial_electrons = wrap_coord(trial_electrons)
    trial_electrons_xy = thetaphi_xy(trial_electrons)

    
    next_lnpsi = v_utils.batch_log_psi(params, model, trial_electrons)
    next_v = v_utils.batch_drift_velocity(params, model, trial_electrons)
    next_d = v_utils.calculate_d_metric_xy(trial_electrons_xy)

    # accepted_idx, acceptance_threshold, log_green_function_forward, log_green_function_backward= calculate_acceptance(key_accept, walker_state.electrons,trial_electrons, walker_state.lnpsi, next_lnpsi, walker_state.v, next_v, walker_state.d_metric, next_d, tau)
    accepted_idx, acceptance_threshold, log_green_function_forward, log_green_function_backward= calculate_acceptance_xy(key_accept, walker_state.electrons_xy, trial_electrons_xy, walker_state.lnpsi, next_lnpsi, walker_state.v, next_v, walker_state.d_metric, next_d, tau)
    start_step_idx = walker_state.dmc_run_step < 1 
    accepted_idx = jnp.where(start_step_idx, jnp.ones_like(walker_state.lnpsi, dtype=bool), accepted_idx)
    # acceptance_threshold = jnp.ones_like(walker_state.lnpsi)
    num_accepted += jnp.sum(accepted_idx)

    # update the walkers according to the acceptance
    next_electrons_xy = jnp.where(accepted_idx[..., None, None], trial_electrons_xy, walker_state.electrons_xy)
    next_electrons = jnp.where(accepted_idx[..., None, None], trial_electrons, walker_state.electrons)
    next_v = jnp.where(accepted_idx[..., None, None], next_v, walker_state.v)
    next_lnpsi = jnp.where(accepted_idx, next_lnpsi, walker_state.lnpsi)
    next_d = jnp.where(accepted_idx[..., None,None], next_d, walker_state.d_metric)

    next_local_energy = v_utils.batch_local_energy(params, system, model, next_electrons)
    # move = jnp.where(accepted_idx[..., None, None], move, jnp.zeros_like(move))
    move = trial_electrons - walker_state.electrons
    xy_move = jnp.where(accepted_idx[..., None, None], xy_move, jnp.zeros_like(move))
 
    
    # total_mean_energy = walker_state.dmc_mean_energy

    next_walker_weights = reweight_walkers(walker_state.weights, walker_state.local_energy, next_local_energy, system.kappa_tau, walker_state.dmc_mean_energy)
    # next_walker_weights = walker_state.weights #without reweighting, it is identical to VMC TODO: verify that it resembles VMC
    next_walker_state = WalkerState(
        electrons=next_electrons,
        electrons_xy=next_electrons_xy,
        v=next_v,
        d_metric=next_d,
        lnpsi=next_lnpsi,
        local_energy=next_local_energy,
        weights=next_walker_weights,
        dmc_mean_energy=walker_state.dmc_mean_energy,
        dmc_run_step=walker_state.dmc_run_step+1
    )

    # print('next dmc_mean E:', next_walker_state.dmc_mean_energy)
    # print(acceptance_threshold)
    # TODO: wrap the output into a debug_info object
    return next_walker_state, key, num_accepted, acceptance_threshold, accepted_idx, walker_state, xy_move, move, log_green_function_forward, log_green_function_backward


def make_dmc_step(system: System, network: LogPsiNetwork, batch_per_device: int, steps: int = 10):
    @jax.jit
    def dmc_step(
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
            walker_state, key, num_accepts, acceptance_threshold, accepted_idx, old_walker_state, xy_move, move, log_green_function_forward, log_green_function_backward = t
            return dmc_update(key, params, system, network, walker_state, num_accepts, tau=system.kappa_tau)
        
        # TODO: fix local energy to a meaningful value
        
        walker_state, key, num_accepts, acceptance_threshold, accepted_idx, old_walker, xy_move, move, log_green_function_forward, log_green_function_backward = lax.fori_loop(
            0, steps, step_fn, (init_walker_state, key, 0, 
                                jnp.ones_like(init_walker_state.lnpsi), jnp.ones_like(init_walker_state.lnpsi, dtype=bool), 
                                init_walker_state, 
                                jnp.zeros_like(init_walker_state.electrons),jnp.zeros_like(init_walker_state.electrons),
                                jnp.zeros_like(init_walker_state.lnpsi),jnp.zeros_like(init_walker_state.lnpsi))  # (walker_state, key, num_accepts)
        )
        print('in dmc_step / step_fn')
        pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
        pmove = constants.pmean(pmove)
        return walker_state, pmove, acceptance_threshold, accepted_idx, old_walker, xy_move, move, log_green_function_forward, log_green_function_backward
    
    return dmc_step


def initialize_walker_state(electrons: jnp.ndarray):
    pass
    