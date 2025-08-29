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


if __name__=="__main__":
    config = Config(network=Network(
            type=NetworkType.dipole_laughlin_v
        ))
    config.seed = 564
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.0001
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    config.optim.iterations = 2000
    config.optim.optimizer = OptimizerName.adam
    config.batch_size = 1024
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    config.log.save_path = "../logs/dipole_laughlin_v_fit"
    config.mcmc.burn_in = 500

    config.log.pretrained_path = None
    vvmc_fit.vvmc_fit(config)
    # vvmc_fit.vvmc_reverse_fit(config)
    config.log.pretrained_path = f"../logs/dipole_laughlin_v_fit/ckpt_00{config.optim.iterations -1}.npz"
    for it in range(2):
        vvmc_fit.vvmc_reverse_fit(config)
    #     vvmc_fit.vvmc_fit(config)
    
