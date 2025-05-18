from deephall import dmc_sample, Config
from deephall.config import Network, NetworkType, System, PsiformerNetwork
from pathlib import Path
from deephall.types import LogPsiNetwork
import jax
import jax.numpy as jnp
import kfac_jax


import pytest
from omegaconf import OmegaConf
from pytest import CaptureFixture


@pytest.fixture
def simple_config():
    config = Config(network=Network(type=NetworkType.laughlin))
    config.seed = 1
    config.system.interaction_strength = 1.0
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.001
    config.system.kappa = 1.0
    config.optim.iterations = 100
    config.batch_size = 198
    config.mcmc.width = 0.3
    
    config.log.pretrained_path = "../logs/laughlin4kappa1.0/ckpt_000999.npz"
    config.log.save_path = "../logs/laughlin4kappa1.0_pytest"
    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 10000
    return config

# def test_initalize_state(simple_config: Config, tmp_path: Path, capsys: CaptureFixture[str]):
#     # TODO: load from pretrained vmc checkpoint
#     # TODO: calculate initial local energy and logpsi and velocity
    
#     log_manager = dmc_sample.LogManager(simple_config)
#     simple_config.log.save_path = str(tmp_path)
#     model = dmc_sample.make_network(simple_config.system, simple_config.network)
#     network = dmc_sample.cast(LogPsiNetwork, model.apply)
#     pmap_mcmc_step, pmove = dmc_sample.setup_mcmc(simple_config, network)
#     print('initial setup_mcmc done')
#     assert simple_config.log.pretrained_path is not None
#     initial_step, (params, walker_state, subkey) = (
#         dmc_sample.initalize_state(simple_config, model)
#     )
#     print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.psi.shape) # [device, batch, Ne, 2]
#     key = jax.random.PRNGKey(simple_config.seed)
#     sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
#     with log_manager.create_writer() as writer:
#         for step in range(simple_config.mcmc.burn_in):
#             sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
#             walker_state, pmove = pmap_mcmc_step(params, walker_state, subkey)
#             # state, stats = training_step(state, subkey)
#             writer.log(
#                 step=str(step),
#                 pmove=f"{pmove[0]:.2f}",
#                 energy=f"{jnp.mean(walker_state.local_energy):.4f}",
#             )

def test_drift_velocity(simple_config: Config, tmp_path: Path, capsys: CaptureFixture[str]):
    # TODO: load from pretrained vmc checkpoint
    # TODO: calculate initial local energy and logpsi and velocity
    
    log_manager = dmc_sample.LogManager(simple_config)
    simple_config.log.save_path = str(tmp_path)
    model = dmc_sample.make_network(simple_config.system, simple_config.network)
    network = dmc_sample.cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmove = dmc_sample.setup_mcmc(simple_config, network)
    print('initial setup_mcmc done')
    assert simple_config.log.pretrained_path is not None
    initial_step, (params, walker_state, subkey) = (
        dmc_sample.initalize_state(simple_config, model)
    )
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.psi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(simple_config.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    with log_manager.create_writer() as writer:
        for step in range(simple_config.mcmc.burn_in):
            print('step', step)
            
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove = pmap_mcmc_step(params, walker_state, subkey)

            print('theta, phi = ', walker_state.electrons[0][0])
            print('Log(psi).real = ',walker_state.psi[0][0])
            print('velocity = ', walker_state.v[0][0])
            print('d-metric = ', walker_state.d_metric[0][0])
            print('E_L = ',walker_state.local_energy[0][0])
            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                energy=f"{jnp.mean(walker_state.local_energy):.4f}",
            )
