# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import NamedTuple, Protocol, TypedDict

from chex import ArrayTree, PRNGKey
from jax import numpy as jnp
from optax import OptState

class WalkerState(NamedTuple):
    electrons: jnp.ndarray = jnp.array([])
    electrons_xy: jnp.ndarray = jnp.array([])
    d_metric: jnp.ndarray = jnp.array([])
    v: jnp.ndarray = jnp.array([])
    lnpsi: jnp.ndarray = jnp.array([])
    local_energy: jnp.ndarray = jnp.array([])
    weights: jnp.ndarray = jnp.array([])
    dmc_mean_energy: jnp.ndarray = jnp.array([])
    dmc_run_step: int = 0

class AngularMomenta(TypedDict):
    """Angular momenta is calculated with kinetic energy."""

    angular_momentum_z: jnp.ndarray
    angular_momentum_z_square: jnp.ndarray
    angular_momentum_square: jnp.ndarray


class OtherObservables(AngularMomenta):
    """Every thing else when calculating local energy."""    
    electrons: jnp.ndarray = jnp.array([])
    electrons_xy: jnp.ndarray = jnp.array([])
    d_metric: jnp.ndarray = jnp.array([])
    v: jnp.ndarray = jnp.array([])
    lnpsi: jnp.ndarray = jnp.array([])
    local_energy: jnp.ndarray = jnp.array([])
    weights: jnp.ndarray = jnp.array([])
    dmc_mean_energy: jnp.ndarray = jnp.array([])
    dmc_run_step: int = 0

    kinetic: jnp.ndarray
    potential: jnp.ndarray


class LossStats(OtherObservables):
    energy: jnp.ndarray
    variance: jnp.ndarray


class CheckpointState(NamedTuple):
    params: ArrayTree
    data: jnp.ndarray
    opt_state: OptState
    mcmc_width: jnp.ndarray


class DMCCheckpointState(NamedTuple):
    params: ArrayTree
    electrons: jnp.ndarray
    electrons_xy: jnp.ndarray
    d_metric: jnp.ndarray
    v: jnp.ndarray
    lnpsi: jnp.ndarray
    local_energy: jnp.ndarray
    weights: jnp.ndarray
    dmc_mean_energy: jnp.ndarray
    dmc_run_step: int
    opt_state: OptState

# kfac optimizer checkpoint state must have all its members be jnp.ndarray with the first two
# dimensions being the number of devices and batch size. That means structures like WalkerState
# cannot be used as is and must be flattened into direct attributes of the DMCCheckpointState.
def get_walker_state(state: DMCCheckpointState) -> WalkerState:
    updates = {}
    for attribute in state._fields:
        if attribute not in ['opt_state', 'params']:
            updates[attribute] = getattr(state, attribute)
    walker_state = WalkerState(**updates)
    return walker_state

def update_from_walker_state(state: DMCCheckpointState, walker_state: WalkerState) -> DMCCheckpointState:
    updates = {
        'params': state.params,
        'opt_state': state.opt_state,
    }
    for attribute in state._fields:
        if attribute not in ['opt_state', 'params']:
            updates[attribute] = getattr(walker_state, attribute)
    return DMCCheckpointState(**updates)

class LocalEnergy(Protocol):
    def __call__(
        self, params: ArrayTree, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, OtherObservables]:
        """Returns the local energy of a Hamiltonian at a configuration.

        Args:
            params: network parameters.
            data: MCMC configuration to evaluate.

        Returns:
            A tuple of
            - Local energy for each walker
            - other observables, such as
                - angular momentum
                - kinetic and potential energy
        """


class LogPsiNetwork(Protocol):
    def __call__(self, params: ArrayTree, data: jnp.ndarray) -> jnp.ndarray:
        pass


class TrainingStep(Protocol):
    def __call__(
        self, state: CheckpointState, key: PRNGKey
    ) -> tuple[CheckpointState, LossStats]:
        pass


class TrainingInit(Protocol):
    def __call__(self, params: ArrayTree, key: PRNGKey, data: jnp.ndarray):
        pass
