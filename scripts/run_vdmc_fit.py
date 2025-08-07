import jax
import jax.numpy as jnp

import numpy as np
from deephall import dmc_sample, Config
from deephall.vdmc import vdmc_fit
from deephall.config import Network, NetworkType, System, PsiformerNetwork,Network, NetworkType, FluxType, FermionicType, PartonNetwork
from pathlib import Path
from deephall.types import LogPsiNetwork
import jax
import jax.numpy as jnp
import kfac_jax
from omegaconf import OmegaConf
import logging
import jax.numpy as jnp
from deephall.train import train


if __name__=="__main__":
    config = Config(network=Network(
            type=NetworkType.super_laughlin_v
        ))
    config.seed = 564
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.0001
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    config.optim.iterations = 10000
    config.batch_size = 128
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    config.log.save_path = "../logs/super_laughlin_v_fit"
    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 1000
    vdmc_fit.vdmc_fit(config)
