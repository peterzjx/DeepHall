from deephall import Config, train
from deephall.config import Network, NetworkType, OptimizerName, FluxType, FermionicType, PartonNetwork
import jax
from datetime import datetime
# jax.config.update("jax_debug_nans", True)


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
for Ne in [4]:
    twoQ = 2*Ne+1
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
        config.system.flux = twoQ
        config.optim.optimizer = OptimizerName.adam
        # config.optim.adam.gradient_accumulation_steps = 10
        config.optim.iterations  = 100000
        config.batch_size = 128
        config.mcmc.width = 0.5
        # config.log.pretrained_path = f"../tasks/psiformer_N{Ne}_kappa{kappa}/ckpt_{config.optim.iterations-1:07d}.npz"
        # config.log.pretrained_path = "../logs/psiformer_laughlin_fit/ckpt_019999.npz"
        # config.log.save_path = f"{timestamp}_pfaffian_{Ne}_kappa_{kappa}"
        config.log.save_path = f"../logs/from_pretrain_psiformer_{Ne}_kappa_{kappa}"
        train(config)
