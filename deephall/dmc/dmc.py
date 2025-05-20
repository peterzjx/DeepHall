
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



def log_green_function_branching(local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
    '''
        local_energy: current local energy
        next_local_energy: next local energy
    '''
    # print('energy shape', next_local_energy.shape)
    print('kappa_tau', kappa_tau)
    print('local_energy', local_energy.shape)
    print('next_local_energy', next_local_energy.shape)
    print('total_mean_energy', total_mean_energy)
    return -kappa_tau * (next_local_energy + local_energy - 2 * total_mean_energy)


def reweight_walkers(weights: jnp.ndarray, local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
    weights = weights * jnp.exp(log_green_function_branching(local_energy, next_local_energy, kappa_tau, total_mean_energy))
    n_walkers = weights.shape[0]
    weights = jnp.sqrt(n_walkers) * weights / jnp.linalg.norm(weights)  # TODO: check if this is correct    
    return weights


def calculate_acceptance(key: PRNGKey, electrons: jnp.ndarray, next_electrons: jnp.ndarray, psi: jnp.ndarray, next_psi: jnp.ndarray, v: jnp.ndarray, next_v: jnp.ndarray, d: float, next_d: float, tau: float):
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

    print('next_electrons', next_electrons.shape)
    print('next_v', next_v.shape)
    print('next_d', next_d.shape)
    print('psi', psi.shape)
    print('next_psi', next_psi.shape)
    log_green_function_forward = log_green_function(electrons, next_electrons, v, d, tau)
    log_green_function_backward = log_green_function(next_electrons, electrons, next_v, next_d, tau)

    acceptance_threshold = (next_psi / psi) ** 2 * jnp.exp(log_green_function_backward - log_green_function_forward)
    walkers_size = acceptance_threshold.shape[0]
    accepted_idx = jax.random.uniform(key, shape=(walkers_size,)) < acceptance_threshold
    return accepted_idx


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
        ) * jnp.sqrt(d_metric) * jnp.sqrt(tau)
        + v * tau
    )
    
    return move


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
    # kappa_tau = system.kappa_tau
    
    move = calculate_move(key_move, walker_state.v, walker_state.d_metric, tau)
    trial_electrons = walker_state.electrons + move
    next_psi = v_utils.batch_log_psi(params, model, trial_electrons)
    next_v = v_utils.batch_drift_velocity(params, model, trial_electrons)
    next_d = v_utils.calculate_d_metric(trial_electrons)

    accepted_idx = calculate_acceptance(key_accept, walker_state.electrons,trial_electrons, walker_state.psi, next_psi, walker_state.v, next_v, walker_state.d_metric, next_d, tau)
    num_accepted += jnp.sum(accepted_idx)

    # update the walkers according to the acceptance
    next_electrons = jnp.where(accepted_idx[..., None, None], trial_electrons, walker_state.electrons)
    next_v = jnp.where(accepted_idx[..., None, None], next_v, walker_state.v)
    next_psi = jnp.where(accepted_idx, next_psi, walker_state.psi)
    next_d = jnp.where(accepted_idx[..., None,None], next_d, walker_state.d_metric)

    next_local_energy = v_utils.batch_local_energy(params, system, model, next_electrons)

    
    # total_mean_energy = walker_state.dmc_mean_energy

    # next_walker_weights = reweight_walkers(walker_state.weights, walker_state.local_energy, next_local_energy, kappa_tau, total_mean_energy)
    next_walker_weights = walker_state.weights #without reweighting, it is identical to VMC TODO: verify that it resembles VMC
    print('current dmc_mean E:', walker_state.dmc_mean_energy)
    next_walker_state = WalkerState(
        electrons=next_electrons,
        v=next_v,
        d_metric=next_d,
        psi=next_psi,
        local_energy=next_local_energy,
        weights=next_walker_weights,
        dmc_mean_energy=walker_state.dmc_mean_energy
    )
    print('next dmc_mean E:', next_walker_state.dmc_mean_energy)
    return next_walker_state, key, num_accepted


def make_dmc_step(system: System, network: LogPsiNetwork, batch_per_device: int, steps: int = 10):
    @jax.jit
    def dmc_step(
        params: ArrayTree, init_walker_state: WalkerState, key: PRNGKey,
    ):
        """Performs a set of DMC steps.

        Args:
        params: parameters to pass to the network.
        data: (batched) DMC configurations to pass to the network.
        key: RNG state.

        Returns:
        (data, pmove), where data is the updated DMC configurations, key the
        updated RNG state and pmove the average probability a move was accepted.
        """
        print('in dmc_step / step_fn')
        def step_fn(i, t):
            walker_state, key, num_accepts = t
            i = i+1
            return dmc_update(key, params, system, network, walker_state, num_accepts, tau=system.kappa_tau)
        
        # TODO: fix local energy to a meaningful value
        
        # walker_state, key, num_accepts = lax.fori_loop(
        #     0, steps, step_fn, (init_walker_state, key, 0)  # (walker_state, key, num_accepts)
        # )

        walker_state, key, num_accepts = step_fn( 0, (init_walker_state, key, 0))
        pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
        pmove = constants.pmean(pmove)
        return walker_state, pmove
    
    return dmc_step


def initialize_walker_state(electrons: jnp.ndarray):
    pass
    