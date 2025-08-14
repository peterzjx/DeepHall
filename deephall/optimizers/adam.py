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

import optax
from chex import PRNGKey
import logging
import jax

from deephall import constants
from deephall.config import OptimizerAdam
from deephall.log import CheckpointState, DMCCheckpointState
from deephall.types import TrainingInit, TrainingStep

logger = logging.getLogger(__name__)


def make_adam_training_step(
    optim_cfg: OptimizerAdam, loss_grad_fn
) -> tuple[TrainingInit, TrainingStep]:
    tx = optax.adam(learning_rate=optim_cfg.lr.schedule)
    gradient_accumulation_steps = optim_cfg.gradient_accumulation_steps
    tx = optax.MultiSteps(tx, gradient_accumulation_steps)


    @constants.pmap
    def init(params, key, data):
        del key, data
        return tx.init(params)

    @constants.pmap
    def step(state: CheckpointState, key: PRNGKey):
        del key
        params, data, opt_state, mcmc_width = state
        stats, grads = loss_grad_fn(params, data)
        updates, opt_state = tx.update(grads, opt_state, params)
        
        # Check if this step actually applied updates (not just accumulated gradients)
        has_updated = tx.has_updated(opt_state)
        
        # Use JAX's debug.print for logging within JAX functions
        import jax.debug
        jax.debug.print("MultiSteps: has_updated={}, gradient_step={}, mini_step={}", 
                       has_updated, opt_state.gradient_step, opt_state.mini_step)
        
        params = optax.apply_updates(params, updates)
        return (CheckpointState(params, data, opt_state, mcmc_width), stats)

    return init, step


def make_adam_training_vvmc_step(
    optim_cfg: OptimizerAdam, loss_grad_fn
) -> tuple[TrainingInit, TrainingStep]:
    # def val_and_grad(params, dat_and_dR):
    #     stats, grads = loss_grad_fn(params, dat_and_dR)
    #     return (stats["energy"], stats), grads

    tx = optax.adam(learning_rate=optim_cfg.lr.schedule)
    gradient_accumulation_steps = optim_cfg.gradient_accumulation_steps
    tx = optax.MultiSteps(tx, gradient_accumulation_steps)


    @constants.pmap
    def init(params, key, data):
        del key, data
        return tx.init(params)

    @constants.pmap
    def step(state: CheckpointState, key: PRNGKey):
        del key
        params, data, opt_state, mcmc_width = state
        stats, grads = loss_grad_fn(params, data)
        updates, opt_state = tx.update(grads, opt_state, params)
        
        # Check if this step actually applied updates (not just accumulated gradients)
        has_updated = tx.has_updated(opt_state)
        
        # Use JAX's debug.print for logging within JAX functions
        import jax.debug
        jax.debug.print("MultiSteps: has_updated={}, gradient_step={}, mini_step={}", 
                       has_updated, opt_state.gradient_step, opt_state.mini_step)
        
        params = optax.apply_updates(params, updates)
        return (CheckpointState(params, data, opt_state, mcmc_width), stats)

    return init, step

def make_adam_training_vvmc_fit_step(
    optim_cfg: OptimizerAdam, loss_grad_fn
) -> tuple[TrainingInit, TrainingStep]:
    # def val_and_grad(params, dat_and_dR):
    #     stats, grads = loss_grad_fn(params, dat_and_dR)
    #     return (stats["energy"], stats), grads

    tx = optax.adam(learning_rate=optim_cfg.lr.schedule)
    gradient_accumulation_steps = optim_cfg.gradient_accumulation_steps
    tx = optax.MultiSteps(tx, gradient_accumulation_steps)


    @constants.pmap
    def init(params, key, data):
        del key, data
        return tx.init(params)

    @constants.pmap
    def step(state: DMCCheckpointState, key: PRNGKey):
        del key
        params, data, electrons_xy, electrons_xy_move, d_metric, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state = state
        stats, grads = loss_grad_fn(params, (electrons_xy, v))
        updates, opt_state = tx.update(grads, opt_state, params)
        
        # Check if this step actually applied updates (not just accumulated gradients)
        # has_updated = tx.has_updated(opt_state)
        
        # Use JAX's debug.print for logging within JAX functions
        # import jax.debug
        # jax.debug.print("MultiSteps: has_updated={}, gradient_step={}, mini_step={}", 
        #                has_updated, opt_state.gradient_step, opt_state.mini_step)
        
        params = optax.apply_updates(params, updates)
        stats['gradient'] = grads
        return (
            DMCCheckpointState(params, data, electrons_xy, electrons_xy_move, d_metric, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state),
            stats
            )

    return init, step