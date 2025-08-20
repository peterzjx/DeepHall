from deephall import Config, train
from deephall.config import Network, NetworkType, FluxType, FermionicType, PartonNetwork
from deephall.config import OptimizerName
import jax
from datetime import datetime
import sys
# jax.config.update("jax_debug_nans", True)


# timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if __name__ == "__main__":
    for Ne in [4]:
        for kappa in [1.0]:
            config = Config(network=Network(
                type=NetworkType.dipole_laughlin,
            ))
            config.system.interaction_strength = kappa
            config.system.nspins = (Ne, 0)
            config.system.flux = 3*(Ne - 1)
            config.optim.optimizer = OptimizerName.none
            config.optim.adam.lr.rate = 0.0000
            config.optim.adam.gradient_accumulation_steps = 10
            config.optim.iterations = 10000
            config.batch_size = 512

            config.mcmc.width = 0.5 
            config.log.save_path = f"../logs/inference_dipole_laughlin_{Ne}_kappa_{kappa}_b{config.batch_size}_mini{config.optim.adam.gradient_accumulation_steps}"
            train(config)
