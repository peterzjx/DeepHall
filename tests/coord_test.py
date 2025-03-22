import jax
import jax.numpy as jnp
import flax.linen as nn

class PairwiseNetwork(nn.Module):
    """Feedforward NN that maps (ri, rj) to a complex number."""
    hidden_dim: int = 64  # Adjustable hidden layer size

    @nn.compact
    def __call__(self, rij):
        """Forward pass: Input shape [batch, Ne, Ne, 4] -> Output [batch, Ne, Ne, 2]"""
        x = nn.Dense(self.hidden_dim)(rij)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)  # Output two values (real & imag)
        return x[..., 0] + 1j * x[..., 1]  # Convert to complex

def extract_pairs(electron):
    """Generate Ne x Ne matrix of (ri, rj) pairs, setting self-pairs to zero."""
    batch, Ne, _ = electron.shape

    # Expand dimensions to create pairwise coordinate tensors
    ri = electron[:, :, None, :]  # Shape: [batch, Ne, 1, 2]
    rj = electron[:, None, :, :]  # Shape: [batch, 1, Ne, 2]

    # Concatenate to form (x_i, y_i, x_j, y_j)
    rij = jnp.concatenate([ri, rj], axis=-1)  # Shape: [batch, Ne, Ne, 4]

    # Zero out self-pairs (i == j)
    mask = jnp.eye(Ne, dtype=electron.dtype)[:, :, None]  # Shape: [Ne, Ne, 1]
    rij = rij * (1 - mask)  # Keep non-diagonal elements, zero out diagonal

    return rij

# Example Usage:
batch, Ne = 32, 10  # 32 configurations, 10 electrons
electron = jax.random.normal(jax.random.PRNGKey(0), (batch, Ne, 2))  # Random electron positions

model = PairwiseNetwork()
params = model.init(jax.random.PRNGKey(42), extract_pairs(electron))  # Initialize model
complex_output = model.apply(params, extract_pairs(electron))  # Forward pass

print(complex_output.shape)  # Expected: [batch, Ne, Ne]
