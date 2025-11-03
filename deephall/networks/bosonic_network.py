import flax.linen as nn
import jax.numpy as jnp
import jax
from itertools import combinations
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
        feature = nn.Dense(32)(uv)  # [..., N, 64]
        feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature = nn.sigmoid(feature)
        feature = nn.Dense(32)(feature)  # [..., N, 128]
        feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature = nn.sigmoid(feature)
        feature = nn.Dense(32)(feature)  # [..., N, 64]
        feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature = nn.sigmoid(feature)
        feature = jnp.max(feature, axis=-2)  # [..., 64]
        flux = nn.Dense(2)(feature)  # [..., 2]
        flux = flux[..., 0] + 1j * flux[..., 1]
        return flux

class SymmetricProductMLPNetwork(nn.Module):
    @nn.compact
    def __call__(self, electrons):
        '''
            electrons: [..., N, 2], theta and phi, both real numbers
            returns: [..., 2], a, b, where a+bi is the value of the wavefunction
        '''
        def get_unordered_pairs(electrons):
            Ne = electrons.shape[-2]

            # Generate all unordered (i < j) index pairs
            pair_indices = jnp.array(list(combinations(range(Ne), 2)))  # shape [num_pairs, 2]
            idx_i, idx_j = pair_indices[:, 0], pair_indices[:, 1]

            # Gather electrons at those indices
            ele_i = electrons[idx_i, :]  # [batch, num_pairs, coord_dim]
            ele_j = electrons[idx_j, :]  # [batch, num_pairs, coord_dim]

            # Concatenate the pairs into one array: [batch, num_pairs, 4]
            ele_pairs = jnp.stack([ele_i, ele_j], axis=-1)
            return ele_pairs
        ele_pairs = get_unordered_pairs(electrons)
        def paired_nn(ele_pair):
            theta, phi = ele_pair[..., 0], ele_pair[..., 1]  # [..., 2], [..., 2]
            uv = jnp.stack([jnp.cos(theta), jnp.cos(phi / 2)], axis= -1)
            # three MLP layers
            feature = nn.Dense(16)(uv)  # [..., N, 64]
            feature = nn.LayerNorm(epsilon=1e-5)(feature)
            feature = nn.sigmoid(feature)
            feature = nn.Dense(32)(feature)  # [..., N, 128]
            feature = nn.LayerNorm(epsilon=1e-5)(feature)
            feature = nn.sigmoid(feature)
            feature = nn.Dense(16)(feature)  # [..., N, 64]
            feature = nn.LayerNorm(epsilon=1e-5)(feature)
            feature = nn.sigmoid(feature)
            feature = jnp.max(feature, axis=-2)  # [..., 64]
            single_flux = nn.Dense(2)(feature)  # [..., 2]
            return single_flux[0]+1j*single_flux[1]
        paired_nn_batched = jax.vmap(paired_nn, in_axes=0, out_axes=0)
        flux_series = paired_nn_batched(ele_pairs)
        flux = jnp.prod(flux_series)
        return flux**2

class SymmetricAttNetwork(nn.Module):

    @nn.compact
    def __call__(self, electrons):
        num_heads = 3
        heads_dim = 256
        '''
            electrons: [..., N, 2], theta and phi, both real numbers
            returns: [..., 2], a, b, where a+bi is the value of the wavefunction
        '''
        theta, phi = electrons[..., 0], electrons[..., 1]  # [..., N], [..., N]
        uv = jnp.stack([jnp.cos(theta), jnp.cos(phi / 2)], axis= -1)
        # three MLP layers

        attention_dim = num_heads * heads_dim
        feature = nn.Dense(attention_dim, use_bias=False)(uv)

        for _ in range(3):
            attn_out = nn.MultiHeadAttention(num_heads=num_heads)(feature)
            feature += nn.Dense(attention_dim, use_bias=False)(attn_out)
            feature = nn.LayerNorm(epsilon=1e-5)(feature)
            feature += nn.tanh(nn.Dense(attention_dim)(feature))
            feature = nn.LayerNorm(epsilon=1e-5)(feature)
        feature = nn.Dense(128)(feature)  # [..., N, 128]

        feature = jnp.max(feature, axis=-2)  # [..., 128]
        flux = nn.Dense(2)(feature)  # [..., 2]
        flux = flux[..., 0] + 1j * flux[..., 1]
        return flux

class SymmetricProductAttNetwork(nn.Module):
    @nn.compact
    def __call__(self, electrons):
        '''
            electrons: [..., N, 2], theta and phi, both real numbers
            returns: [..., 2], a, b, where a+bi is the value of the wavefunction
        '''
        def get_unordered_pairs(electrons):
            Ne = electrons.shape[-2]

            # Generate all unordered (i < j) index pairs
            pair_indices = jnp.array(list(combinations(range(Ne), 2)))  # shape [num_pairs, 2]
            idx_i, idx_j = pair_indices[:, 0], pair_indices[:, 1]

            # Gather electrons at those indices
            ele_i = electrons[idx_i, :]  # [batch, num_pairs, coord_dim]
            ele_j = electrons[idx_j, :]  # [batch, num_pairs, coord_dim]

            # Concatenate the pairs into one array: [batch, num_pairs, 4]
            ele_pairs = jnp.stack([ele_i, ele_j], axis=-1)
            return ele_pairs
        ele_pairs = get_unordered_pairs(electrons)
        
        def paired_nn(ele_pair):
            num_heads = 3
            heads_dim = 32
            theta, phi = electrons[..., 0], electrons[..., 1]  # [..., N], [..., N]
            uv = jnp.stack([jnp.cos(theta), jnp.cos(phi / 2)], axis= -1)
            # three MLP layers

            attention_dim = num_heads * heads_dim
            feature = nn.Dense(attention_dim, use_bias=False)(uv)

            for _ in range(3):
                attn_out = nn.MultiHeadAttention(num_heads=num_heads)(feature)
                feature += nn.Dense(attention_dim, use_bias=False)(attn_out)
                feature = nn.LayerNorm(epsilon=1e-5)(feature)
                feature += nn.tanh(nn.Dense(attention_dim)(feature))
                feature = nn.LayerNorm(epsilon=1e-5)(feature)
            feature = nn.Dense(64)(feature)  # [..., N, 128]

            feature = jnp.max(feature, axis=-2)  # [..., 128]
            flux = nn.Dense(2)(feature)  # [..., 2]
            flux = flux[..., 0] + 1j * flux[..., 1]
            return flux
        
        paired_nn_batched = jax.vmap(paired_nn, in_axes=0, out_axes=0)
        flux_series = paired_nn_batched(ele_pairs)
        flux = jnp.prod(flux_series)
        return flux**2
