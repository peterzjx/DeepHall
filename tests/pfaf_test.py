from deephall import Config, train
from deephall.config import Network, NetworkType, System,PsiformerNetwork
import jax
# jax.config.update("jax_debug_nans", True)


for Ne in [4,6,8]:
    for kappa in [0.3, 0.5, 2, 4]:
        config = Config(network=Network(type=NetworkType.pfaffian))
        config.system.interaction_strength = kappa
        config.system.nspins = (Ne, 0)
        config.system.flux = 2*Ne+1
        config.optim.iterations = 5000
        config.batch_size = 128
        config.mcmc.width = 0.3
        config.log.save_path = "pfaffian"+str(Ne)+"kappa"+str(kappa)
        train(config)

        config = Config(network=Network(type=NetworkType.psiformer))
        config.system.interaction_strength = kappa
        config.system.nspins = (Ne, 0)
        config.system.flux = 2*Ne+1
        config.optim.iterations = 5000
        config.batch_size = 128
        config.mcmc.width = 0.3
        config.log.save_path = "psiformer"+str(Ne)+"kappa"+str(kappa)
        train(config)
