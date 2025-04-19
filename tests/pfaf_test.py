from deephall import Config, train
from deephall.config import Network, NetworkType, System,PsiformerNetwork
import jax
from datetime import datetime
# jax.config.update("jax_debug_nans", True)


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
for Ne in [6]:
    for kappa in [0.5]:
        config = Config(network=Network(type=NetworkType.pfaffian))
        config.system.interaction_strength = kappa
        config.system.nspins = (Ne, 0)
        config.system.flux = 2*Ne+1
        config.optim.iterations  = 10000
        config.batch_size = 128
        config.mcmc.width = 0.3
        config.log.pretrained_path = f"20250418221957_pfaffian_4_kappa_0.5"
        config.log.save_path = f"{timestamp}_pfaffian_{Ne}_kappa_{kappa}"
        train(config)

        # config = Config(network=Network(type=NetworkType.psiformer))
        # config.system.interaction_strength = kappa
        # config.system.nspins = (Ne, 0)
        # config.system.flux = 2*Ne+1
        # config.optim.iterations = 10000
        # config.batch_size = 128
        # config.mcmc.width = 0.3
        # config.log.save_path = f"{timestamp}_psiformer_{Ne}_kappa_{kappa}"
        # train(config)
