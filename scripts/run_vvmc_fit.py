from deephall import Config
from deephall.vvmc import vvmc_fit
from deephall.config import Network, NetworkType, OptimizerName
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
    config.optim.iterations = 5000
    config.optim.optimizer = OptimizerName.adam
    config.batch_size = 4096
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    # config.log.pretrained_path = "../logs/super_laughlin_v_fit/ckpt_099999.npz"
    config.log.save_path = "../logs/super_laughlin_v_fit"
    config.mcmc.burn_in = 5000

    # config.log.pretrained_path = None
    # vvmc_fit.vvmc_fit(config)
    config.log.pretrained_path = "../logs/super_laughlin_v_fit/ckpt_004999.npz"
    for it in range(10):
        vvmc_fit.vvmc_reverse_fit(config)
        vvmc_fit.vvmc_fit(config)
    
