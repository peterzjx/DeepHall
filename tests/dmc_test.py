from deephall import dmc_sample, Config
from deephall.config import Network, NetworkType, System, PsiformerNetwork
from pathlib import Path
from deephall.types import LogPsiNetwork
import jax
import kfac_jax


import pytest
from omegaconf import OmegaConf
from pytest import CaptureFixture


        

@pytest.fixture
def simple_config():
    config = Config(network=Network(type=NetworkType.laughlin))
    config.system.interaction_strength = 1.0
    config.system.nspins = (5, 0)
    config.system.flux = 12
    config.optim.iterations = 10000
    config.batch_size = 792
    config.mcmc.width = 0.3
    
    config.log.pretrained_path = "../logs/laughlin4kappa1.0/ckpt_000999.npz"
    config.log.save_path = "../logs/laughlin4kappa1.0_pytest"
    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 10
    return config

def test_initalize_state(simple_config: Config, tmp_path: Path, capsys: CaptureFixture[str]):
    log_manager = dmc_sample.LogManager(simple_config)
    simple_config.log.save_path = str(tmp_path)
    model = dmc_sample.make_network(simple_config.system, simple_config.network)
    network = dmc_sample.cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmoves = dmc_sample.setup_mcmc(simple_config, network)
    assert simple_config.log.pretrained_path is not None
    initial_step, (params, data, mcmc_width) = (
        dmc_sample.initalize_state(simple_config, model)
    )
    # _, (params, _, _) = (
    #     log_manager.try_load_pretrained_checkpoint()
    # )
    key = jax.random.PRNGKey(simple_config.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    for _ in range(simple_config.mcmc.burn_in):
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        data, pmove = pmap_mcmc_step(params, data, subkey)