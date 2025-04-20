import flax.linen as nn
import jax.numpy as jnp

class SymmetricMLPNetwork(nn.Module):

    @nn.compact
    def __call__(self, electrons):
        '''
            electrons: [..., N, 2], theta and phi, both real numbers
            returns: [..., 2], a, b, where a+bi is the value of the wavefunction
        '''
        theta, phi = electrons[..., 0], electrons[..., 1]  # [..., N], [..., N]
        uv = jnp.stack([jnp.cos(theta), jnp.cos(phi / 2)], axis= -1)
        # three MLP layers
        feature = nn.Dense(128)(uv)  # [..., N, 64]
        feature = nn.Dense(256)(feature)  # [..., N, 128]
        feature = nn.Dense(64)(feature)  # [..., N, 64]

        feature = jnp.max(feature, axis=-2)  # [..., 64]
        flux = nn.Dense(2)(feature)  # [..., 2]
        return flux

class SymmetricNetwork(nn.Module):

    @nn.compact
    def __call__(self, electrons):
        num_heads = 3
        heads_dim = 128
        '''
            electrons: [..., N, 2], theta and phi, both real numbers
            returns: [..., 2], a, b, where a+bi is the value of the wavefunction
        '''
        theta, phi = electrons[..., 0], electrons[..., 1]  # [..., N], [..., N]
        uv = jnp.stack([jnp.cos(theta), jnp.cos(phi / 2)], axis= -1)
        # three MLP layers

        attention_dim = num_heads * heads_dim
        feature = nn.Dense(attention_dim, use_bias=False)(uv)
        attn_out = nn.MultiHeadAttention(num_heads=num_heads)(feature)
        feature += nn.Dense(attention_dim, use_bias=False)(attn_out)
        feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature += nn.tanh(nn.Dense(attention_dim)(feature))
        feature = nn.LayerNorm(epsilon=1e-5)(feature)

        feature = nn.Dense(128)(feature)  # [..., N, 128]
        feature = nn.Dense(64)(feature)  # [..., N, 64]

        feature = jnp.max(feature, axis=-2)  # [..., 64]
        flux = nn.Dense(2)(feature)  # [..., 2]
        return flux