import jax
import jax.numpy as jnp
from flax import linen as nn


# --- Basic Attention Block ---
class SelfAttentionBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        # --- Attention sublayer ---
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            use_bias=True,
        )(x)
        x = residual + x  # residual add

        # --- Feed-forward sublayer ---
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.d_model * self.mlp_ratio)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        x = residual + x  # residual add

        return x


# --- Stage 1: Global feature from all electrons ---
class SystemFeatureExtractor(nn.Module):
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 1

    @nn.compact
    def __call__(self, electrons):  # [N, 2]
        x = nn.Dense(self.d_model)(electrons)  # [N, d_model]
        for _ in range(self.n_layers):
            x = SelfAttentionBlock(self.d_model, self.n_heads)(x)
        # Pool across all electrons -> global system feature
        system_feature = jnp.mean(x, axis=0)  # [d_model]
        return system_feature


# --- Stage 2: Combine (ei, ej, system) and use CLS ---
class PairSystemFeatureNet(nn.Module):
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2

    @nn.compact
    def __call__(self, pair, electrons):
        """
        pair: [2, 2]   -> two specific electrons (ei, ej)
        electrons: [N, 2] -> all electrons
        returns: [d_model] system+pair feature
        """
        # Step 1. Get global system feature from all electrons
        system_feature = SystemFeatureExtractor(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
        )(electrons)  # [d_model]
        # Step 2. Embed the pair electrons
        pair_embed = nn.Dense(self.d_model)(pair)  # [2, d_model]
        
        # Step 3. Stack: [system_feature, ei, ej]
        seq = jnp.vstack([system_feature[None, :], pair_embed])  # [3, d_model]
        seq = seq.flatten()
        return seq
    
        # # # Step 4. Add CLS token
        # # cls = self.param("cls", nn.initializers.normal(), (1, self.d_model))
        # # seq = jnp.concatenate([cls, seq], axis=0)  # [4, d_model]

        # # Step 5. Pass through attention layers #TODO: Check layer norm/ batch norm
        # for _ in range(self.n_layers):
        #     seq = SelfAttentionBlock(self.d_model, self.n_heads)(seq)

        # # Step 6. Return CLS as next-level feature
        # print('in PairSystemFeatureNet: seq ', seq.shape)
        # return seq[0]  # [d_model]
