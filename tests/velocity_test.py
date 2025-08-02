from pathlib import Path
import jax
import jax.numpy as jnp
import logging
import time
from deephall import Config, vdmc_sample
from deephall.types import LogPsiNetwork, get_walker_state
from deephall.velocity_networks import LaughlinVelocity
from deephall.config import NetworkType
from deephall.hamiltonian import local_v_energy
import kfac_jax

import pytest
from pytest import CaptureFixture

def thetaphi_xy(electron_thetaphi: jnp.ndarray):
    theta = electron_thetaphi[..., 0]
    phi = electron_thetaphi[..., 1]
    x = jnp.cos(phi) / jnp.tan(theta / 2)
    y = jnp.sin(phi) / jnp.tan(theta / 2)
    electron_xy = jnp.stack([x, y], axis=-1)
    return electron_xy


@pytest.fixture
def simple_cfg():
    config = Config()
    config.network.type = NetworkType.laughlin_v
    config.seed = 1
    config.system.nspins = (4, 0)
    config.system.flux = 9
    config.system.interaction_strength = 1.0
    config.optim.iterations = 100
    config.batch_size = 6
    config.mcmc.burn_in = 200
    config.mcmc.iteration = 0
    config.initial_energy = config.system.nspins[0] * 0.5 + 0.467 * config.system.nspins[0] * config.system.interaction_strength
    config.log.initial_energy = False
    config.log.save_path = "../logs/test_logs"
    return config

def test_LnPsi_v(simple_cfg: Config):
    model = LaughlinVelocity(nspins=simple_cfg.system.nspins, flux=simple_cfg.system.flux)

    key = jax.random.PRNGKey(42)
    Ne = sum(simple_cfg.system.nspins)

    theta = jnp.array([1.6856816, 2.4018655, 1.5067337, 0.42490557])
    phi = jnp.array([-2.8268971, -2.2615817, -0.6118226, -2.640316])
    electrons = jnp.stack([theta, phi], axis=-1)  # shape: (Ne, 2)
    electrons_xy = thetaphi_xy(electrons)
    # Apply model
    variables = model.init(key, electrons_xy)
    v_xy = model.apply(variables, electrons_xy)
    
    v_real = jnp.real(v_xy)
    v_imag = jnp.imag(v_xy)
    # v_real_target = jnp.array([[-1.73629, 2.31639], [5.06336, 3.15901], [1.03646,  1.73465], [-0.229379,  - 0.0176224]])
    # v_imag_target = jnp.array([[-0.932702, -5.98708], [-0.821033, 3.13067], [0.844674, 4.71262], [0.909061, -1.85621]])
    
    

    move = (
        jax.random.normal(
            key=key,
            shape=electrons.shape
        ) * 0.03
    )
    print('move = ', move)
    electrons_next = electrons + move
    next_electrons_xy = thetaphi_xy(electrons_next)
    next_v_xy = model.apply(variables, next_electrons_xy)

    dR = next_electrons_xy - electrons_xy
    _2F = jnp.real(v_xy + next_v_xy)
    
    wfn_ratio = jnp.exp(jnp.vdot(dR, _2F))
    print(v_xy)
    
    print(next_v_xy)
    

    print('ratio = ', wfn_ratio)
    
    # acceptance_threshold = acceptance_threshold * next_metric / metric

# def test_laughlin_velocity(simple_cfg: Config):
#     # Parameters

#     model = LaughlinVelocity(nspins=simple_cfg.system.nspins, flux=simple_cfg.system.flux)

#     key = jax.random.PRNGKey(42)
#     Ne = sum(simple_cfg.system.nspins)

#     theta = jnp.array([1.6856816, 2.4018655, 1.5067337, 0.42490557])
#     phi = jnp.array([-2.8268971, -2.2615817, -0.6118226, -2.640316])
#     electrons = jnp.stack([theta, phi], axis=-1)  # shape: (Ne, 2)
#     electrons_xy = thetaphi_xy(electrons)
#     print('electrons_xy', electrons_xy.shape)
#     # Apply model
#     variables = model.init(key, electrons_xy)
#     F = model.apply(variables, electrons_xy)
#     FF = jnp.sum(F * F, axis = -1)
#     print('FF', FF, jnp.sum(FF))
#     v_real = jnp.real(F)
#     v_imag = jnp.imag(F)
#     v_real_target = jnp.array([[-1.73629, 2.31639], [5.06336, 3.15901], [1.03646,  1.73465], [-0.229379,  - 0.0176224]])
#     v_imag_target = jnp.array([[-0.932702, -5.98708], [-0.821033, 3.13067], [0.844674, 4.71262], [0.909061, -1.85621]])
#     assert jnp.all(jnp.abs(v_real - v_real_target)< 1e-4)
#     assert jnp.all(jnp.abs(v_imag - v_imag_target)< 1e-4)
#     print("Input electrons (theta, phi):")
#     print(electrons)
#     print("\nComputed velocities")
#     print(v_real)
#     print(v_imag)

# def test_vdmc(simple_cfg: Config):
#     # --- Set up logging ---
#     # logger = logging.getLogger(__name__)
#     # logger.setLevel(logging.INFO)

#     # # File handler
#     # file_handler = logging.FileHandler(simple_cfg.log.save_path + '/status_loggings.log')
#     # file_handler.setLevel(logging.INFO)

#     # # Console handler
#     # console_handler = logging.StreamHandler()
#     # console_handler.setLevel(logging.INFO)

#     # # Formatter
#     # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     # file_handler.setFormatter(formatter)
#     # console_handler.setFormatter(formatter)

#     # # Add handlers to the logger
#     # logger.addHandler(file_handler)
#     # logger.addHandler(console_handler)
# # ==================================================================

    
#     log_manager = vdmc_sample.LogManager(simple_cfg)
#     model = vdmc_sample.make_v_network(simple_cfg.system, simple_cfg.network)
#     network = vdmc_sample.cast(LogPsiNetwork, model.apply)
#     pmap_mcmc_step, pmove = vdmc_sample.setup_mcmc(simple_cfg, network)
#     print('initial setup_mcmc done', pmap_mcmc_step)
#     _, state = (
#         vdmc_sample.initalize_state(simple_cfg, model)
#     )
#     walker_state = get_walker_state(state)
#     params = state.params

#     # print('Initial walker_state shape:', walker_state.electrons.shape, walker_state.v.shape, walker_state.lnpsi.shape) # [device, batch, Ne, 2]
#     key = jax.random.PRNGKey(simple_cfg.seed)
#     sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
#     energy_history = jnp.ones(1000)*simple_cfg.initial_energy
    
#     start = time.time()

    
#     for step in range(simple_cfg.mcmc.burn_in):
#         print("Step burn in #", step)
#         sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
#         walker_state, pmove = pmap_mcmc_step(params, walker_state, subkey)
#         print('p = ',pmove, walker_state.dmc_run_step)
#         local_mean_energy = vdmc_sample.weighted_mean_energy(walker_state=walker_state)
#         energy_history.at[step % len(energy_history)].set(local_mean_energy)
#         walker_state = vdmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=5000, use_external_energy=True, external_energy=local_mean_energy)
#     walker_state = vdmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=1, reweight_interval=1, use_external_energy=True, external_energy=local_mean_energy)            

    
#     with log_manager.create_writer() as writer:
#         renormal_interval = 100
#         energy_update_interval = 1000
        
#         for step in range(simple_cfg.mcmc.iteration):
            
        
#             sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
#             walker_state, pmove  = pmap_mcmc_step(params, walker_state, subkey)
#             # Ensure all device work is finished before continuing
#             # walker_state = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, walker_state)
            
#             local_mean_energy = vdmc_sample.weighted_mean_energy(walker_state=walker_state)     
#             print("Step ", step)
#             print('E = ', local_mean_energy, ' p = ', jnp.mean(pmove))  
#             energy_history.at[step % len(energy_history)].set(local_mean_energy)
#             hist_mean_energy = jnp.mean(energy_history)
            
#             walker_state = vdmc_sample.update_mean_energy(walker_state=walker_state,step=step,update_interval=energy_update_interval,reweight_interval=renormal_interval,use_external_energy=True, external_energy=local_mean_energy)
        
#             writer.log(
#                 step=str(step),
#                 pmove=f"{pmove[0]:.2f}",
#                 local_energy=f"{local_mean_energy:.6f}",
#                 dmc_mean_energy=f"{jnp.mean(walker_state.dmc_mean_energy):.6f}",
#                 history_mean_energy=f"{hist_mean_energy:.6f}",
#                 weight_max=f"{jnp.max(walker_state.weights):.6f}",
#                 weight_min=f"{jnp.min(walker_state.weights):.6f}",
#                 weight_std=f"{jnp.std(walker_state.weights):.6f}" 
#             )
            
#             if step%renormal_interval==0 and (jnp.min(walker_state.weights)<0.01 or jnp.max(walker_state.weights)>5.0) and renormal_interval>10:
#                 renormal_interval = renormal_interval - 1
#             assert renormal_interval>=10