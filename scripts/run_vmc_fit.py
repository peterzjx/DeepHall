from deephall import Config
from deephall.vmc_pretrain import vmc_fit
from deephall.config import Network, NetworkType, OptimizerName, FluxType, FermionicType, PartonNetwork
from pathlib import Path
from deephall.types import LogPsiNetwork
import jax
import jax.numpy as jnp
import kfac_jax
from omegaconf import OmegaConf
import logging
import jax.numpy as jnp


if __name__=="__main__":
    Ne = 4
    twoQ = 9

    laughlin_config = Config(network=Network(
                type=NetworkType.laughlin
            ))
    laughlin_config.system.interaction_strength = 1.0
    laughlin_config.system.nspins = (Ne, 0)
    laughlin_config.system.flux = twoQ
    laughlin_config.optim.optimizer = OptimizerName.adam
    laughlin_config.optim.adam.gradient_accumulation_steps = 4
    laughlin_config.mcmc.use_vmc_pretrain = True
    
    config = Config(network=Network(
            type=NetworkType.psiformer,
            # parton=PartonNetwork(
            #     fermionic_type=FermionicType.pfaffian,
            #     flux_type=FluxType.original_jastrow
            # )
        ))
    config.seed = 564
    config.system.nspins = (Ne, 0)
    config.system.flux = twoQ
    config.system.tau = 0.0001
    config.system.interaction_strength = 1.0
    config.optim.iterations = 20000
    config.optim.optimizer = OptimizerName.adam
    config.optim.adam.gradient_accumulation_steps = 4
    config.optim.adam.lr.rate = 1e-5
    config.batch_size = 1024
    config.mcmc.use_vmc_pretrain = True
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    # config.log.pretrained_path = "../logs/psiformer_laughlin_fit/ckpt_001352.npz"
    config.log.save_path = "../logs/psiformer_laughlin_fit"
    config.mcmc.burn_in = 500

    vmc_fit.vmc_fit(laughlin_config, config)