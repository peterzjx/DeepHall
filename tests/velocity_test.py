from pathlib import Path
import jax
import jax.numpy as jnp
import logging
import time
from deephall import Config, vvmc_sample
from deephall.types import LogPsiNetwork, get_walker_state
from deephall.velocity_networks import LaughlinVelocity, SuperLaughlinVelocity
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
    config.batch_size = 48
    config.mcmc.burn_in = 100
    config.mcmc.iteration = 100
    config.initial_energy = config.system.nspins[0] * 0.5 + 0.467 * config.system.nspins[0] * config.system.interaction_strength
    config.log.initial_energy = False
    config.log.save_path = "../logs/test_logs"
    return config

def test_TwoBodyFermionV(simple_cfg: Config):
    from deephall.velocity_networks.super_laughlin_v import MLP, TwoBodyVelocity
    wfn = TwoBodyVelocity([16, 16, 4])
    z = jnp.array([[1.6856816, 2.4018655], [1.5067337, 0.42490557]])
    ex_z = jnp.flip(z, axis=0)
    key = jax.random.PRNGKey(0)
    params = wfn.init(key, z)
    y = wfn.apply(params, z)
    yp = wfn.apply(params, ex_z)
    print(z)
    print(ex_z)
    print('v(z)', y)
    print('v(z\')', yp)
    # assert jnp.all(y == - yp)

def test_super_Laughlin_v(simple_cfg: Config):
    model = SuperLaughlinVelocity(nspins=simple_cfg.system.nspins, flux=simple_cfg.system.flux, hidden_features=(32,32))

    key = jax.random.PRNGKey(42)
    Ne = sum(simple_cfg.system.nspins)

    theta = jnp.array([1.6856816, 2.4018655, 1.5067337, 0.42490557])
    phi = jnp.array([-2.8268971, -2.2615817, -0.6118226, -2.640316])
    electrons = jnp.stack([theta, phi], axis=-1)  # shape: (Ne, 2)
    electrons_xy = thetaphi_xy(electrons)
    # Apply model
    variables = model.init(key, electrons_xy)
    v_xy = model.apply(variables, electrons_xy)
    
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
    element_wise = dR * _2F
    # Sum over last two dimensions
    dot_product = jnp.sum(element_wise, axis=(-1, -2))
    wfn_ratio = jnp.exp(dot_product)
    print('shape ', dR.shape, _2F.shape)
    print(v_xy)
    print(next_v_xy)
    print('ratio = ', wfn_ratio)
    

# def test_Laughlin_accept_ratio(simple_cfg: Config):
#     model = LaughlinVelocity(nspins=simple_cfg.system.nspins, flux=simple_cfg.system.flux)

#     key = jax.random.PRNGKey(42)
#     Ne = sum(simple_cfg.system.nspins)

#     theta = jnp.array([1.6856816, 2.4018655, 1.5067337, 0.42490557])
#     phi = jnp.array([-2.8268971, -2.2615817, -0.6118226, -2.640316])
#     electrons = jnp.stack([theta, phi], axis=-1)  # shape: (Ne, 2)
#     electrons_xy = thetaphi_xy(electrons)
#     # Apply model
#     variables = model.init(key, electrons_xy)
#     v_xy = model.apply(variables, electrons_xy)
    
#     move = (
#         jax.random.normal(
#             key=key,
#             shape=electrons.shape
#         ) * 0.03
#     )
#     print('move = ', move)
#     electrons_next = electrons + move
#     next_electrons_xy = thetaphi_xy(electrons_next)
#     next_v_xy = model.apply(variables, next_electrons_xy)

#     dR = next_electrons_xy - electrons_xy
#     _2F = jnp.real(v_xy + next_v_xy)
#     element_wise = dR * _2F
#     # Sum over last two dimensions
#     dot_product = jnp.sum(element_wise, axis=(-1, -2))
#     wfn_ratio = jnp.exp(dot_product)
#     print('shape ', dR.shape, _2F.shape)
#     print(v_xy)
#     print(next_v_xy)
#     print('ratio = ', wfn_ratio)
#     assert jnp.abs(wfn_ratio - 1.0224) <1e-4

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