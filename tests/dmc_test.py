import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
import numpy as np
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
    config.seed = 564
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.tau = 0.001
    config.system.interaction_strength = 1.0
    config.system.kappa_tau = config.system.tau * config.system.interaction_strength
    config.optim.iterations = 100
    config.batch_size = 6
    config.mcmc.width = 0.3
    
    config.log.pretrained_path = "../logs/laughlin4kappa1.0/ckpt_000999.npz"
    config.log.save_path = "../logs/laughlin4kappa1.0_pytest"
    config.mcmc.use_dmc = True
    config.mcmc.burn_in = 2001
    return config

def test_drift_velocity(simple_config: Config, tmp_path: Path, capsys: CaptureFixture[str]):
    # TODO: load from pretrained vmc checkpoint
    # TODO: calculate initial local energy and logpsi and velocity
    import logging
    import jax.numpy as jnp

    # --- Set up logging ---
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(simple_config.log.save_path + '/status_loggings.log')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
# ==================================================================


    log_manager = dmc_sample.LogManager(simple_config)
    simple_config.log.save_path = str(tmp_path)
    model = dmc_sample.make_network(simple_config.system, simple_config.network)
    network = dmc_sample.cast(LogPsiNetwork, model.apply)
    pmap_mcmc_step, pmove = dmc_sample.setup_mcmc(simple_config, network)
    print('initial setup_mcmc done', pmap_mcmc_step)
    assert simple_config.log.pretrained_path is not None
    initial_step, (params, walker_state, subkey) = (
        dmc_sample.initalize_state(simple_config, model)
    )
    print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
    key = jax.random.PRNGKey(simple_config.seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    dxdy_hist = []
    with log_manager.create_writer() as writer:
        for step in range(simple_config.mcmc.burn_in):
            print('step', step)
            
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            walker_state, pmove, acceptance_threhold, accepted_idx, old_walker, xy_move, move, log_green_function_forward, log_green_function_backward  = pmap_mcmc_step(params, walker_state, subkey)
            walker_state = dmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=200)
            
            print('theta, phi = ', old_walker.electrons[0])
            print('xy =', old_walker.electrons_xy[0])
            print('d-metrix = ', old_walker.d_metric[0])
            print('velocity = ', old_walker.v[0][0])
            print('Log(psi).real = ',old_walker.lnpsi[0])
            print("theta\', phi\' = ", walker_state.electrons[0])
            print('xy\' =', walker_state.electrons_xy[0])
            print('d-metrix\' = ', walker_state.d_metric[0])
            print('velocity\' = ', walker_state.v[0][0])
            print('Log(psi)\'.real = ',walker_state.lnpsi[0])
            print('move = ', move[0])
            print('xy_move = ', xy_move[0])
            print('accept', acceptance_threhold[0])
            print('accept idx', accepted_idx[0])
            # xy = dmc_sample.dmc.thetaphi_xy(old_walker.electrons)
            # dxdy_hist.append(xy)
            # dxdy_hist.append(move)
            dxdy_hist.append(old_walker.electrons)
            
            # if jnp.any(jnp.abs(walker_state.v) >50):
            #     mask = jnp.abs(walker_state.v) >50
            #     indices = jnp.argwhere(mask)
            #     print('err coord back', old_walker.electrons[indices][0])
            #     print('err coord back Log(psi) = ',old_walker.lnpsi[indices][0])
            #     print('err v_back', old_walker.v[indices][0])
                
            #     print('err coord', walker_state.electrons[indices][0])
            #     print('err coord back Log(psi) = ',walker_state.lnpsi[indices][0])
            #     print('err v', walker_state.v[indices][0])
                
                
            #     print('err lnG_forward', log_green_function_forward[indices][0])
            #     print('err lnG_back', log_green_function_backward[indices][0])

            #     print('err accept', acceptance_threhold[indices][0])
            #     break

            # xy = dmc_sample.dmc.thetaphi_xy(walker_state.electrons)
            # print('x, y = ', xy[0])
            # print('Log(psi).real = ',walker_state.lnpsi[0])
            # print('velocity = ', walker_state.v[0][0])
            # print('d-metric = ', walker_state.d_metric[0][0])
            # print('E_L = ',walker_state.local_energy[0][0])
            # print('accceptance ratio', acceptance_threhold[0])
            # print('dmc_mean_energy_shape', walker_state.dmc_mean_energy.shape, walker_state.weights.shape, walker_state.local_energy.shape)
            # logger.info("theta_phi: %s, xy: %s, Ln|Psi|: %s", walker_state.electrons[0].tolist(), xy[0].tolist(),walker_state.lnpsi[0].tolist())
            # logger.info("theta_phi: %s, xy: %s, Ln|Psi|: %s", old_walker.electrons[0].tolist(), xy[0].tolist(),walker_state.lnpsi[0].tolist())
            writer.log(
                step=str(step),
                pmove=f"{pmove[0]:.2f}",
                energy=f"{jnp.mean(walker_state.local_energy):.4f}",
                dmc_mean_energy=f"{jnp.mean(walker_state.dmc_mean_energy):.4f}"
            )
    dxdy_hist_array = jnp.stack(dxdy_hist)
    dxdy_hist_array = jnp.reshape(dxdy_hist_array, (-1, 2))
    from matplotlib import pyplot as plt
    plt.axis('equal')
    plt.plot(dxdy_hist_array[0::4,0], dxdy_hist_array[0::4,1], '.', color='red')
    plt.plot(dxdy_hist_array[1::4,0], dxdy_hist_array[1::4,1], '.', color='blue')
    plt.plot(dxdy_hist_array[2::4,0], dxdy_hist_array[2::4,1], '.', color='green')
    plt.plot(dxdy_hist_array[3::4,0], dxdy_hist_array[3::4,1], '.', color='orange')
    plt.show()
    print(dxdy_hist_array.shape)

