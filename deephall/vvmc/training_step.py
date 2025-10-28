import optax
from chex import PRNGKey
import logging
from typing import cast

import jax
import kfac_jax
from jax import numpy as jnp

from deephall import constants
from deephall.config import OptimizerAdam, OptimizerKfac
from deephall.log import DMCCheckpointState
from deephall.types import TrainingInit, TrainingStep
from deephall.vvmc.loss import make_vvmc_fit_loss_fn
from deephall.types import LogPsiNetwork
from deephall.config import Config, OptimizerName
from deephall.loss import LossStats, make_vvmc_loss_fn

### VVMC fit step ###

def make_training_step_vvmc_fit(
    cfg: Config, network: LogPsiNetwork
) -> tuple[TrainingInit, TrainingStep]:
    loss_grad_fn = make_vvmc_fit_loss_fn(network) # loss_grad_fn(param, [electrons_xy, v])
    assert cfg.optim.optimizer == OptimizerName.adam
    return make_adam_training_step_vvmc_fit(cfg.optim.adam, loss_grad_fn)


def make_adam_training_step_vvmc_fit(
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
    def step(state: DMCCheckpointState, key: PRNGKey):
        del key
        params, data, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state = state
        stats, grads = loss_grad_fn(params, (electrons_xy, v))
        updates, opt_state = tx.update(grads, opt_state, params)
        
        params = optax.apply_updates(params, updates)
        stats['gradient'] = grads
        return (
            DMCCheckpointState(params, data, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state),
            stats
            )

    return init, step

### VVMC training step ###

def make_training_step_vvmc(
    cfg: Config, network: LogPsiNetwork, graph_patterns
) -> tuple[TrainingInit, TrainingStep]:
    """Create training step for VVMC energy training."""
    loss_grad_fn = make_vvmc_loss_fn(network, cfg.system)
    
    if cfg.optim.optimizer == OptimizerName.none:
        return make_inference_vvmc_step(loss_grad_fn)
    if cfg.optim.optimizer == OptimizerName.adam:
        return make_adam_training_step_vvmc(cfg.optim.adam, loss_grad_fn)
    if cfg.optim.optimizer == OptimizerName.kfac:
        return make_kfac_training_step_vvmc(cfg.optim.kfac, loss_grad_fn, graph_patterns)
    
    raise ValueError(f"Optimizer {cfg.optim.optimizer} is not implemented for VVMC!")


def make_adam_training_step_vvmc(
    optim_cfg: OptimizerAdam, loss_grad_fn
) -> tuple[TrainingInit, TrainingStep]:

    tx = optax.adam(learning_rate=optim_cfg.lr.schedule)
    gradient_accumulation_steps = optim_cfg.gradient_accumulation_steps
    jax.debug.print(f"multi step ={gradient_accumulation_steps}")
    tx = optax.MultiSteps(tx, gradient_accumulation_steps)


    @constants.pmap
    def init(params, key, xy_and_dR):
        del key, xy_and_dR
        return tx.init(params)

    @constants.pmap
    def step(state: DMCCheckpointState, key: PRNGKey):
        del key
        params, electron, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state = state
        stats, grads = loss_grad_fn(params, (electrons_xy, electrons_xy_move))
        updates, opt_state = tx.update(grads, opt_state, params)
        
        params = optax.apply_updates(params, updates)
        return (
            DMCCheckpointState(params, electron, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state),
            stats
            )

    return init, step


def make_inference_step_vvmc(loss_grad_fn) -> tuple[TrainingInit, TrainingStep]:
    @constants.pmap
    def init(params, key, data):
        del params, key, data
        return None

    @constants.pmap
    def step(state: DMCCheckpointState, key: PRNGKey):
        del key
        params, electron, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state = state
        stats, _ = loss_grad_fn(params, (electrons_xy, electrons_xy_move))
        return (
            DMCCheckpointState(params, electron, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state),
            stats
            )

    return init, step


def make_kfac_training_step_vvmc(
    optim_cfg: OptimizerKfac, loss_grad_fn, graph_patterns
) -> tuple[TrainingInit, TrainingStep]:
    def val_and_grad(params, xy_and_dX):
        stats, grads = loss_grad_fn(params, xy_and_dX)
        return (stats["energy"], stats), grads

    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=0.0,
        norm_constraint=1e-3,
        value_func_has_aux=True,
        learning_rate_schedule=optim_cfg.lr.schedule,
        curvature_ema=0.95,
        inverse_update_period=1,
        min_damping=1e-4,
        num_burnin_steps=0,
        register_only_generic=False,
        estimation_mode="fisher_exact",
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=graph_patterns,
        ),
    )
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(1e-3))

    def init(params, key, data):
        return optimizer.init(params, key, data)

    def step(state: DMCCheckpointState, key: PRNGKey): #VVMC version
        params, electrons, electrons_xy, electron_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state = state
        params, opt_state, *_, stats = optimizer.step(
            params=params,
            state=opt_state,
            rng=key,
            batch=(electrons_xy, electron_xy_move),  # Pass both electrons and weights as a tuple
            momentum=shared_mom,
            damping=shared_damping,
        )
        return (
            DMCCheckpointState(params, electrons, electrons_xy, electron_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state),
            cast(LossStats, stats["aux"]),
        )

    return init, step
