from deephall import Config, train
from deephall.config import Network, NetworkType, System,PsiformerNetwork
import jax
# jax.config.update("jax_debug_nans", True)



config = Config(network=Network(type=NetworkType.pfaffian))
config.system.nspins = (4, 0)
config.system.flux = 5
config.system.interaction_strength = 1.0
config.optim.iterations = 50000
config.batch_size = 256
config.mcmc.width = 0.3
train(config)
