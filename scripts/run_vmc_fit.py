from deephall import Config
from deephall.vmc_pretrain import vmc_fit
from deephall.config import Network, NetworkType, OptimizerName, PartonNetwork, FermionicType, FluxType


if __name__=="__main__":
    Ne = 3
    twoQ = 6

    laughlin_config = Config(network=Network(
                type=NetworkType.laughlin
            ))
    laughlin_config.system.interaction_strength = 1.0
    laughlin_config.system.nspins = (Ne, 0)
    laughlin_config.system.flux = twoQ
    laughlin_config.optim.optimizer = OptimizerName.adam
    laughlin_config.optim.adam.gradient_accumulation_steps = 4
    laughlin_config.mcmc.use_vmc_pretrain = True
    
    # config = Config(network=Network(
    #         type=NetworkType.psiformer
    #     ))
    config = Config(network=Network(
            type=NetworkType.modulus_phase,
        ))
    # config = Config(network=Network(
    #         type=NetworkType.parton,
    #         parton=PartonNetwork(
    #             fermionic_type=FermionicType.pfaffian,
    #             flux_type=FluxType.original_jastrow
    #         )
    #     ))
    config.seed = 128
    config.system.nspins = (Ne, 0)
    config.system.flux = twoQ
    config.system.tau = 0.0001
    config.system.interaction_strength = 1.0
    config.optim.iterations = 20000
    config.optim.optimizer = OptimizerName.adam
    config.optim.adam.gradient_accumulation_steps = 2
    config.optim.adam.lr.rate = 1e-5
    config.batch_size = 4096
    config.mcmc.use_vmc_pretrain = True
    config.mcmc.width = 0.3
    config.initial_energy = 0.0
    
    config.log.save_step_interval = 100
    config.log.save_coords = True
    config.log.save_coords_step_interval = 100
    # config.log.pretrained_path = "../logs/psiformer_laughlin_fit/ckpt_001352.npz"
    config.log.save_path = "logs/pretrain_test"
    config.mcmc.burn_in = 500

    vmc_fit.vmc_fit(laughlin_config, config)