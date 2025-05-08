
from typing import NamedTuple
from flax import linen as nn
import jax
from jax import lax
from jax import numpy as jnp
from chex import ArrayTree, PRNGKey
import deephall.dmc.velocity_utils as v_utils
from deephall import constants
from deephall.types import WalkerState, LogPsiNetwork




def log_green_function_branching(local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
    '''
        local_energy: current local energy
        next_local_energy: next local energy
    '''
    return -kappa_tau * (next_local_energy + local_energy - 2 * total_mean_energy)


def reweight_walkers(weights: jnp.ndarray, local_energy: jnp.ndarray, next_local_energy: jnp.ndarray, kappa_tau: float, total_mean_energy: float):
    weights = weights * jnp.exp(log_green_function_branching(local_energy, next_local_energy, kappa_tau, total_mean_energy))
    n_walkers = weights.shape[0]
    weights = jnp.sqrt(n_walkers) * weights / jnp.linalg.norm(weights)  # TODO: check if this is correct    
    return weights

def calculate_d_metric():
    pass


def calculate_acceptance(key: PRNGKey, electrons: jnp.ndarray, next_electrons: jnp.ndarray, psi: jnp.ndarray, next_psi: jnp.ndarray, v: jnp.ndarray, next_v: jnp.ndarray, d: float, next_d: float, tau: float):
    '''
        key: jax.random.PRNGKey
        electrons: electrons coordinates
        next_electrons: next electrons coordinates
        psi: current psi
        next_psi: next psi
        v: current velocity
        next_v: next velocity
        d0: d metric
        tau: time step
    '''

    log_green_function_forward = log_green_function(electrons, next_electrons, v, d, tau)
    log_green_function_backward = log_green_function(next_electrons, electrons, next_v, next_d, tau)

    acceptance_threshold = (next_psi / psi) ** 2 * jnp.exp(log_green_function_backward - log_green_function_forward)
    walkers_size = acceptance_threshold.shape[0]
    accepted_idx = jax.random.uniform(key, shape=(walkers_size,)) < acceptance_threshold
    return accepted_idx


def calculate_move(key: PRNGKey, v: jnp.ndarray, d0: float, tau: float):
    '''
        key: jax.random.PRNGKey
        v: velocity
        d0: d metric
        tau: time step
    '''
    print('vshape', v.shape)
    print('d0', d0.shape)
    move = (
        jax.random.normal(
            key=key,
            shape=v.shape
            # mean=jnp.zeros_like(v), #TODO: Check if mean is needed. This is not defined in normal()
            # std=jnp.sqrt(d0) * jnp.sqrt(tau)  # isotropic along x and y
        ) * jnp.sqrt(d0) * jnp.sqrt(tau)
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
    displacement = electrons_to - electrons_from - v_from * d * tau
    squared_distances = jnp.sum(displacement ** 2, axis=-1)  # (n_walkers, n_electrons)

    expo = -0.5 * squared_distances / (d * tau)  # (n_walkers, n_electrons)

    # Sum over electrons and add log term
    return jnp.sum(expo, axis=1) - 2.0 * jnp.sum(jnp.log(d), axis=1)


def step(key: PRNGKey, params: ArrayTree, model: nn.Module, walker_state: WalkerState, tau: float):
    '''
        key: jax.random.PRNGKey
        params: network parameters
        model: psi model, callable
        walker_state: current walker state
        tau: time step
    '''
    key, key_move, key_accept = jax.random.split(key, 3)
    d0 = v_utils.calculate_d_metric(walker_state.electrons)
    move = calculate_move(key_move, walker_state.v, d0, tau)
    trial_electrons = walker_state.electrons + move
    next_v = v_utils.drift_velocity(params, model, trial_electrons)
    next_psi = v_utils.psi(params, model, trial_electrons)

    accepted_idx = calculate_acceptance(key_accept, walker_state.psi, next_psi, walker_state.v, next_v)
    num_accepted = jnp.sum(accepted_idx)

    # update the walkers according to the acceptance
    next_electrons = jnp.where(accepted_idx, trial_electrons, walker_state.electrons)
    next_v = jnp.where(accepted_idx, next_v, walker_state.v)
    next_psi = jnp.where(accepted_idx, next_psi, walker_state.psi)

    next_local_energy = v_utils.local_energy(params, model, next_electrons)

    kappa_tau = 1.0  # TODO: get from params
    total_mean_energy = 0.0  # TODO: get from params

    next_walker_weights = reweight_walkers(walker_state.weights, walker_state.local_energy, next_local_energy, kappa_tau, total_mean_energy)

    next_walker_state = WalkerState(
        electrons=next_electrons,
        v=next_v,
        psi=next_psi,
        local_energy=next_local_energy,
        weights=next_walker_weights
    )

    return next_walker_state, key, next_local_energy, num_accepted


def make_dmc_step(batch_network: LogPsiNetwork, batch_per_device: int, steps: int = 10):
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

        def step_fn(i, walker_state):
            return step(key, params, batch_network, walker_state, tau=0.01)

        walker_state, key, _, num_accepts = lax.fori_loop(
            0, steps, step_fn, init_walker_state
        )
        pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
        pmove = constants.pmean(pmove)
        return walker_state, pmove
    
    return dmc_step


def initialize_walker_state(electrons: jnp.ndarray):
    pass
