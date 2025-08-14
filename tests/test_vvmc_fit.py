import jax
from deephall import Config
from deephall import optimizers
from deephall.config import Network, NetworkType, OptimizerName
from deephall.types import get_walker_state, update_from_walker_state
from pathlib import Path
from deephall.types import LogPsiNetwork
from deephall.loss import LossMode, make_loss_fn
from deephall.velocity_networks import make_v_network
import jax
import kfac_jax
from omegaconf import OmegaConf
import logging
from deephall.train import train
from typing import cast
import pytest
from pytest import CaptureFixture

import logging
from deephall.log import LogManager, init_logging
from deephall import vvmc_sample
import time
logger = logging.getLogger("deephall")
@pytest.fixture
def simple_cfg():
    config = Config(network=Network(
            type=NetworkType.super_laughlin_v
        ))
    config.seed = 564
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.0001
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    config.optim.iterations = 100
    config.optim.optimizer = OptimizerName.adam
    config.batch_size = 48
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    config.log.save_path = "../logs/super_laughlin_v_fit"
    config.mcmc.burn_in = 100
    return config

def get_laughlin_cfg(cfg: Config):
    config = Config()
    config.network.type = NetworkType.laughlin_v
    config.seed = 1
    config.system.nspins = cfg.system.nspins
    config.system.flux = cfg.system.flux
    config.system.interaction_strength = cfg.system.interaction_strength
    config.optim.iterations = cfg.optim.iterations
    config.batch_size = cfg.batch_size
    config.mcmc.burn_in = cfg.mcmc.burn_in
    config.initial_energy = cfg.initial_energy
    config.log.initial_energy = False
    config.log.save_path = cfg.log.save_path
    return config

def test_vvmc_fit(simple_cfg: Config):
    from deephall.vvmc import vvmc_fit
    vvmc_fit.vvmc_fit(simple_cfg)