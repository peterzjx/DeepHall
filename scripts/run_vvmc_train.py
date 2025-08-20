import jax
import jax.numpy as jnp

import numpy as np
from deephall import dmc_sample, Config
from deephall import constants, mcmc, optimizers
from deephall.config import Network, NetworkType, System, PsiformerNetwork,Network, NetworkType, FluxType, FermionicType, PartonNetwork, OptimizerName
from deephall.types import CheckpointState, DMCCheckpointState, WalkerState, get_walker_state, update_from_walker_state
from pathlib import Path
from deephall.types import LogPsiNetwork
from deephall.loss import LossMode, make_loss_fn
from deephall.velocity_networks import make_v_network
from deephall.vvmc import vvmc_train
import jax
import jax.numpy as jnp
import kfac_jax
from omegaconf import OmegaConf
import logging
import jax.numpy as jnp
from deephall.train import train
from typing import cast
import pytest
from pytest import CaptureFixture

import logging
from deephall.log import LogManager, init_logging
from deephall import vvmc_sample
logger = logging.getLogger("deephall")

if __name__=="__main__":
    config = Config(network=Network(
            type=NetworkType.super_laughlin_v
        ))
    config.seed = 564
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.interaction_strength = 6.0
    config.optim.optimizer = OptimizerName.adam
    config.optim.adam.lr.rate = 1e-4
    config.optim.adam.gradient_accumulation_steps = 100
    # config.optim.optimizer = OptimizerName.kfac
    # config.optim.kfac.lr.rate = 1e-6

    config.optim.iterations = 50000
    config.batch_size = 4096
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    # config.log.pretrained_path = "../logs/super_laughlin_v_fit/ckpt_001999.npz"
    config.log.pretrained_path = "../logs/super_laughlin_v_fit/ckpt_001999.npz"
    config.log.save_path = f"../logs/super_laughlin_v_train_k{config.system.interaction_strength}"
    config.mcmc.burn_in = 2000

    vvmc_train.vvmc_train(config)
    