from pathlib import Path
import jax
import jax.numpy as jnp
import logging
import time
from deephall import Config, vvmc_sample
from deephall.types import LogPsiNetwork, get_walker_state
from deephall.velocity_networks import DipoleLaughlinVelocity
from deephall.velocity_networks.attention_block import SystemFeatureExtractor
from deephall.config import NetworkType
from deephall.hamiltonian import local_v_energy
import kfac_jax
import numpy as np
import pytest
from pytest import CaptureFixture

jax.config.update("jax_enable_x64", True)
@pytest.fixture
def cfg():
    config = Config()
    config.network.type = NetworkType.dipole_laughlin_v
    config.seed = 1
    config.system.nspins = (3, 0)
    config.system.flux = 6
    config.system.interaction_strength = 1.0
    config.optim.iterations = 100
    config.batch_size = 48
    config.mcmc.burn_in = 100
    config.mcmc.iteration = 100
    # config.initial_energy = config.system.nspins[0] * 0.5 + 0.467 * config.system.nspins[0] * config.system.interaction_strength
    config.log.initial_energy = False
    config.log.save_path = "../logs/test_logs"
    return config
def test_permutation_invariance():
    key = jax.random.key(0)

    # Example: 5 electrons, each with 2D coordinates
    electrons = jax.random.normal(key, (5, 2))

    model = SystemFeatureExtractor(d_model=64, n_heads=4, n_layers=2)

    # Init parameters
    variables = model.init(key, electrons)

    # Reference output
    out_ref = model.apply(variables, electrons)

    # Test multiple permutations
    max_diff = 0.0
    for i in range(20):   # try 20 random permutations
        perm = np.random.permutation(electrons.shape[0])
        electrons_perm = electrons[perm]
        out_perm = model.apply(variables, electrons_perm)
        diff = float(jnp.max(jnp.abs(out_ref - out_perm)))
        max_diff = max(max_diff, diff)

    print("Max difference over 50 permutations:", max_diff)
    assert max_diff < 1e-5, f"Permutation invariance broken! (max diff {max_diff})"

def test_exchange_sym(cfg: Config):
    key = jax.random.PRNGKey(0)
    # velocity_model = DipoleLaughlinVelocity(cfg.system.nspins, cfg.system.flux, (32,32))
    from deephall.velocity_networks import make_v_network
    velocity_model = make_v_network(cfg.system, cfg.network)
    electron_xy = jnp.transpose(jnp.array([[1.6856816, 2.4018655, 1.5067337],[-2.8268971, -2.2615817, -0.6118226]]))
    # electron_xy = electron_xy[...,0]+1j*electron_xy[...,1]
    variables = velocity_model.init(key, electron_xy)
    v0 = velocity_model.apply(variables, electron_xy)
    def swap_electrons(seq, i, j):
        """
        Swap rows i and j in a sequence array seq.
        seq: [Ne, d] array
        i, j: indices to swap
        returns: new sequence with rows i and j swapped
        """
        seq_new = seq.at[i].set(seq[j])  # put row j into position i
        seq_new = seq_new.at[j].set(seq[i])  # put row i into position j
        return seq_new
    z01 = swap_electrons(electron_xy, 0, 1)
    v01 = velocity_model.apply(variables, z01)

    print('z: ')
    print(electron_xy)
    print(z01)
    print('\n')
    print(v0)
    print(v01)