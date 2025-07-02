from deephall import Config, train
from deephall.config import Network, NetworkType, FluxType, FermionicType, PartonNetwork
import jax
from datetime import datetime
# jax.config.update("jax_debug_nans", True)


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
for Ne in [8]:
    for kappa in [1.0]:
        config = Config(network=Network(
            type=NetworkType.psiformer,
            # parton=PartonNetwork(
            #     fermionic_type=FermionicType.pfaffian,
            #     flux_type=FluxType.symmetric_mlp_network
            # )
        ))
        config.system.interaction_strength = kappa
        config.system.nspins = (Ne, 0)
        config.system.flux = (5*Ne-8) // 2
        config.optim.iterations  = 200000
        config.batch_size = 128
        config.mcmc.width = 0.3
        config.log.save_path = f"../logs/psiformer25_{Ne}_kappa_{kappa}"
        train(config)
