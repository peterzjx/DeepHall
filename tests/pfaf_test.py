from deephall import Config, train
from deephall.config import Network, NetworkType, System,PsiformerNetwork
import jax
# jax.config.update("jax_debug_nans", True)


for Ne in [8,]:
    config = Config(network=Network(type=NetworkType.pfaffian))
    config.system.nspins = (Ne, 0)
    config.system.flux = 2*Ne+1
    config.system.interaction_strength = 1.0
    config.optim.iterations = 5000
    config.batch_size = 256
    config.mcmc.width = 0.3
    config.log.save_path = "pfaffian"+str(Ne)
    train(config)

    config = Config(network=Network(type=NetworkType.psiformer))
    config.system.nspins = (Ne, 0)
    config.system.flux = 2*Ne+1
    config.system.interaction_strength = 1.0
    config.optim.iterations = 5000
    config.batch_size = 256
    config.mcmc.width = 0.3
    config.log.save_path = "psiformer"+str(Ne)
    train(config)
