from deephall import Config, train

config = Config()
config.system.nspins = (4, 0)
config.system.flux = 9
config.system.interaction_strength = 1.0
config.optim.iterations = 50000
config.network.psiformer.num_layers = 2
config.batch_size = 512

train(config)