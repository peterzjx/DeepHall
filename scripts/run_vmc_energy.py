from deephall import Config, train
from deephall.config import Network, NetworkType, FluxType, FermionicType, PartonNetwork
import jax
from datetime import datetime
# jax.config.update("jax_debug_nans", True)


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
for Ne in [4]:
    for kappa in [1.0]:
        config = Config(network=Network(
            type=NetworkType.modulus_phase,
        ))
        config.system.interaction_strength = kappa
        config.system.nspins = (Ne, 0)
        config.system.flux = 3*(Ne-1)
        config.optim.iterations  = 10000
        config.batch_size = 512
        config.mcmc.width = 0.5
        config.log.pretrained_path = "logs/pretrain_test/ckpt_007104.npz"
        config.log.save_path = f"logs/modulus_phase_energy_N{Ne}_kappa{kappa}_{timestamp}"
        train(config)