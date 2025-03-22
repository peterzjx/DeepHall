import jax.numpy as jnp

def reconstruct_antisymmetric(n, values):
    i, j = jnp.triu_indices(n, k=1)  # Upper triangle indices
    M = jnp.zeros((n, n))  # Initialize zero matrix
    M = M.at[i, j].set(values)  # Set upper triangle
    return M - M.T  # Enforce antisymmetry directly

# Example usage
n = 3
values = jnp.array([2, -3, 4])  # Extracted values from upper triangle
M_reconstructed = reconstruct_antisymmetric(n, values)

print(M_reconstructed)
# Expected output:
# [[ 0  2 -3]
#  [-2  0  4]
#  [ 3 -4  0]]
