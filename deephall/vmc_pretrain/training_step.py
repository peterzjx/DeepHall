import optax
from chex import PRNGKey
import logging

from deephall import constants
from deephall.config import OptimizerAdam
from deephall.log import DMCCheckpointState
from deephall.types import TrainingInit, TrainingStep
from deephall.vmc_pretrain.loss import make_vmc_fit_loss_fn
from deephall.types import LogPsiNetwork
from deephall.config import Config, OptimizerName

def make_training_step_vmc_fit(
    cfg: Config, network: LogPsiNetwork
) -> tuple[TrainingInit, TrainingStep]:
    loss_grad_fn = make_vmc_fit_loss_fn(network) # loss_grad_fn(param, [electrons, v])
    assert cfg.optim.optimizer == OptimizerName.adam
    return make_adam_training_step_vmc_fit(cfg.optim.adam, loss_grad_fn)


def make_adam_training_step_vmc_fit(
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
        params, electons, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state = state
        stats, grads = loss_grad_fn(params, (electons, lnpsi))
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        stats['gradient'] = grads
        return (
            DMCCheckpointState(params, electons, electrons_xy, electrons_xy_move, d_metric, last_v, v, lnpsi, local_energy, weights, dmc_mean_energy, dmc_run_step, opt_state),
            stats
            )

    return init, step