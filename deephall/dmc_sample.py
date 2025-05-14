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

import logging
import signal
import sys
import time
from argparse import ArgumentParser
from typing import cast

import jax
import kfac_jax
import numpy as np
from chex import PRNGKey
from flax import linen as nn
from jax import numpy as jnp
from omegaconf import OmegaConf

from deephall import constants, mcmc, optimizers
from deephall.config import Config, OptimizerName
from deephall.log import LogManager, init_logging
from deephall.types import CheckpointState, DMCCheckpointState
from deephall.loss import LossMode, make_loss_fn
from deephall.networks import make_network
from deephall.types import LogPsiNetwork
from deephall.dmc import dmc
from deephall.dmc.dmc import WalkerState
from chex import ArrayTree
import deephall.dmc.velocity_utils as v_utils

logger = logging.getLogger("deephall")


def init_guess(key: PRNGKey, batch: int, nelec: int):
    """Create uniform samples on the sphere.

    Args:
        key: random key.
        batch: number of samples to generate.
        nelec: number of electrons.

    Returns:
        Electron coordinates of shape [batch, nelec, 2]
    """
    key1, key2 = jax.random.split(key)
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.stack([theta, phi], axis=-1)


def initalize_state(cfg: Config, model: nn.Module):
    key_data, key_params = jax.random.split(jax.random.PRNGKey(cfg.seed))
    coords = init_guess(key_data, cfg.batch_size, sum(cfg.system.nspins))
    coords = coords.reshape((jax.device_count(), -1, *coords.shape[-2:]))
    params = kfac_jax.utils.replicate_all_local_devices(
        model.init(key_params, coords[0, 0])
    )
    # TODO: calculate initial logpsi and v, after loading a pretrained params
    # logpsi_0 = v_utils.log_psi(params, model, coords)  # [nwalkers,]
    # v_0 = v_utils.drift_velocity(params, model, coords)

    v_0 = jnp.ones((coords.shape[0],))
    logpsi_0 = jnp.ones((coords.shape[0],))
    walker_state = WalkerState(
        electrons=coords,
        v=v_0,
        psi=logpsi_0,
        local_energy=jnp.zeros_like(logpsi_0),  # TODO: calculate local energy
        weights=jnp.ones_like(logpsi_0)
    )

    # initial_step, (params, walker_state, opt_state)
    return 0, DMCCheckpointState(params, walker_state, None)


def setup_mcmc(cfg: Config, network: LogPsiNetwork):
    if cfg.mcmc.use_dmc:
        mcmc_step = dmc.make_dmc_step(
            cfg.system,
            network,
            batch_per_device=cfg.batch_size // jax.device_count(),
            steps=cfg.mcmc.steps,
        )
    else:
        pass
    pmap_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    return pmap_mcmc_step, pmoves


def update_walker_state_from_pretrained(cfg: Config, model: nn.Module, params: ArrayTree, walker_state: WalkerState):
    psi = v_utils.psi(params, model, walker_state.electrons)
    v = v_utils.drift_velocity(params, model, walker_state.electrons)
    energy = v_utils.local_energy(params, model, walker_state.electrons)
    return WalkerState(
        electrons=walker_state.electrons,
        v=v,
        psi=psi,
        local_energy=energy,
        weights=walker_state.weights
    )

# def sample_test(cfg: Config):
#     init_logging()
#     log_manager = LogManager(cfg)
#     model = make_network(cfg.system, cfg.network)
#     network = cast(LogPsiNetwork, model.apply)
#     pmap_mcmc_step, pmoves = setup_mcmc(cfg, network)
#     opt_init, training_step = optimizers.make_optimizer_step(cfg, network)

#     key = jax.random.PRNGKey(cfg.seed)
#     sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

#     # if cfg.log.pretrained_path is not None:
#     #     initial_step, (params, data, opt_state, mcmc_width) = (
#     #         initalize_state(cfg, model)
#     #     )
#     #     _, (params, _, opt_state, _) = (
#     #         log_manager.try_load_pretrained_checkpoint()
#     #     )
#     # else:
#     #     initial_step, (params, data, opt_state, mcmc_width) = (
#     #         log_manager.try_restore_checkpoint() or initalize_state(cfg, model)
#     #     )

#     # TODO: load pretrained model from vmc checkpoint

#     initial_step, (params, walker_state, opt_state) = (
#         initalize_state(cfg, model)
#     )
#     # NOTE: walker state after this step only contains electron coordinates and weights
    
#     walker_state = update_walker_state_from_pretrained(cfg, model, params, walker_state)
#     # updated velocity, local energy, psi

#     if (
#         cfg.optim.optimizer == OptimizerName.none
#         and cfg.log.restore_path is not None
#         and cfg.log.restore_path != cfg.log.save_path
#     ):  # Reset steps because inference run is another run
#         initial_step = 0

#     if opt_state is None:
#         sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
#         opt_state = opt_init(params, subkey, walker_state.electrons)

#     logger.info("Start VMC with %s JAX devices", jax.device_count())

#     if initial_step == 0:
#         print('walker_state', walker_state)
#         for _ in range(cfg.mcmc.burn_in):
#             sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
#             walker_state, pmove = pmap_mcmc_step(params, walker_state, subkey)
#         logger.info("Burn in MCMC complete")
#         if cfg.log.initial_energy:
#             # Logging inital energy is helpful for debugging. If we have initial energy
#             # but have error in training, it's probably optimizer's fault
#             initial_stats, _ = constants.pmap(
#                 make_loss_fn(network, cfg.system, LossMode.ENERGY_DIFF)
#             )(params, walker_state)
#             logger.info("Initial energy: %s", initial_stats["energy"][0].real)


# def cli(argv: list[str] | None = None) -> None:
#     parser = ArgumentParser(
#         prog="deephall",
#         description="Simulating the fractional quantum Hall effect (FQHE) with "
#         "neural network variational Monte Carlo.",
#     )
#     parser.add_argument(
#         "dotlist", help="path.to.key=value pairs for configuration", nargs="*"
#     )
#     parser.add_argument("--yml", help="config YML file to merge")
#     args = parser.parse_args(argv or sys.argv[1:] or ["--help"])

#     config = OmegaConf.structured(Config)
#     if args.yml:
#         config = OmegaConf.merge(config, OmegaConf.load(args.yml))
#     config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.dotlist))
#     sample_test(Config.from_dict(config))


# if __name__ == "__main__":
#     cli()
