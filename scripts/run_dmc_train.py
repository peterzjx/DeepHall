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
    config = Config(network=Network(
            type=NetworkType.parton,
            parton=PartonNetwork(
                fermionic_type=FermionicType.pfaffian,
                flux_type=FluxType.symmetric_mlp_network
            )
            # type=NetworkType.psiformer
        ))
    config.seed = 564
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.001
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    config.optim.iterations = 10000
    config.batch_size = 128
    config.mcmc.width = 0.3

    config.log.save_step_interval = 100
    config.log.pretrained_path = "../logs/pfaf_4_kappa_1.0_dmc/ckpt_000709.npz"
    config.log.save_path = "../logs/pfaf_4_kappa_1.0_dmc"
    # config.log.pretrained_path = "../logs/psiformer_4_kappa_1.0/ckpt_012999.npz"
    # config.log.save_path = "../logs/psiformer_4_kappa_1.0_dmc"
    # config.log.pretrained_path = "../logs/laughlin_4_kappa_1.0/ckpt_003884.npz"
    # config.log.save_path = "../logs/laughlin_4_kappa_1.0_dmc"
    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 200

    # dmc_train.dmc_train(config)
    train(config)
