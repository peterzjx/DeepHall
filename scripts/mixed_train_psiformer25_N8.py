import jax
import jax.numpy as jnp

import numpy as np
from deephall import dmc_sample, Config
from deephall.dmc import dmc_train
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
    Ne = 8
    config = Config(network=Network(
            # type=NetworkType.parton,
            # parton=PartonNetwork(
            #     fermionic_type=FermionicType.pfaffian,
            #     flux_type=FluxType.symmetric_mlp_network
            # )
            type=NetworkType.psiformer
        ))
    config.seed = 564
    config.system.nspins = (Ne, 0)
    config.system.flux = (5*Ne-8) // 2
    config.system.tau = 0.0005
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    
    config.batch_size = 128
    config.mcmc.width = 0.3
    config.initial_energy = Ne * 0.5 + 0.934 * Ne * config.system.interaction_strength
    config.log.save_step_interval = 100
    config.mcmc.burn_in = 3000
    dmc_iteration = 6000
    vmc_iteration = 80000
    for iteration in range(2):
        
        
        config.optim.iterations = vmc_iteration
        config.mcmc.use_dmc = False
        config.log.pretrained_path = f"../logs/psiformer25_{Ne}_kappa_1.0_dmc/ckpt_00{dmc_iteration-1}.npz"
        config.log.save_path = f"../logs/psiformer25_{Ne}_kappa_1.0_vmc"
        train(config)

        config.seed = config.seed + iteration * 12307
        config.optim.iterations = dmc_iteration
        config.mcmc.use_dmc = True
        config.log.pretrained_path = f"../logs/psiformer25_{Ne}_kappa_1.0_vmc/ckpt_0{vmc_iteration-1}.npz"
        config.log.save_path = f"../logs/psiformer25_{Ne}_kappa_1.0_dmc"
        dmc_train.dmc_train(config)
