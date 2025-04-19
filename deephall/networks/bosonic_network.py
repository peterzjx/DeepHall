import flax.linen as nn
import jax.numpy as jnp
class SymmetricNetwork(nn.Module):

    @nn.compact
    def __call__(self, electrons):
        '''
            electrons: [..., N, 2], theta and phi, both real numbers
            returns: [..., 2], a, b, where a+bi is the value of the wavefunction
        '''
        # three MLP layers
        electrons = nn.Dense(64)(electrons)  # [..., N, 64]
        electrons = nn.Dense(128)(electrons)  # [..., N, 128]
        electrons = nn.Dense(64)(electrons)  # [..., N, 64]

        electrons = jnp.max(electrons, axis=-2)  # [..., 64]
        flux = nn.Dense(2)(electrons)  # [..., 2]
        return flux