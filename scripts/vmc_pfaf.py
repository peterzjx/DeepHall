from deephall import Config, train
from deephall.config import Network, NetworkType, FluxType, FermionicType, PartonNetwork
from deephall.config import OptimizerName
import jax
from datetime import datetime
import sys
# jax.config.update("jax_debug_nans", True)


# timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    for Ne in [4]:
        for kappa in [1.0]:
            config = Config(network=Network(
                type=NetworkType.parton,
                parton=PartonNetwork(
                    fermionic_type=FermionicType.pfaffian,
                    flux_type=FluxType.symmetric_product_mlp_network
                )
            ))
            config.system.interaction_strength = kappa
            config.system.nspins = (Ne, 0)
            config.system.flux = 2*Ne+1
            # config.optim.optimizer = OptimizerName.kfac
            config.optim.iterations  = 20000
            config.batch_size = batch_size
            config.mcmc.width = 0.5 
            # config.log.pretrained_path = f"20250418221957_pfaffian_4_kappa_0.5"
            # config.log.save_path = f"{timestamp}_pfaffian_{Ne}_kappa_{kappa}"
            # config.log.save_path = f"../logs/{timestamp}_pfaf_{Ne}_kappa_{kappa}"
            config.log.save_path = f"../logs/pfaf_{Ne}_kappa_{kappa}_prod_mlp_pole_{batch_size}"
            train(config)

            # config = Config(network=Network(type=NetworkType.psiformer))
            # config.system.interaction_strength = kappa
            # config.system.nspins = (Ne, 0)
            # config.system.flux = 2*Ne+1
            # config.optim.iterations = 8000
            # config.batch_size = 128
            # config.mcmc.width = 0.3
            # # config.log.save_path = f"{timestamp}_psiformer_{Ne}_kappa_{kappa}"
            # config.log.save_path = f"psiformer_{Ne}_kappa_{kappa}"
            # train(config)
