import jax
import jax.numpy as jnp
from flax import linen as nn
from deephall.velocity_networks.attention_block import PairSystemFeatureNet
def main():
    key = jax.random.PRNGKey(0)

    N = 6
    pair = jax.random.normal(key, (2, 2))        # [ei, ej]
    electrons = jax.random.normal(key, (N, 2))   # [e1...eN]

    model = PairSystemFeatureNet(d_model=32, n_heads=4, n_layers=2)
    variables = model.init(key, pair, electrons)
    feature = model.apply(variables, pair, electrons)

    print("Final feature shape:", feature.shape)  # [32]


if __name__ == "__main__":
    main()
