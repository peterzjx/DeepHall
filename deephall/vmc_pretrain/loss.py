import jax
import jax.numpy as jnp
from typing import Callable
from deephall.types import ArrayTree, LogPsiNetwork

def make_vmc_fit_loss_fn(
    network: LogPsiNetwork
) -> Callable[[ArrayTree, tuple[jnp.ndarray, jnp.ndarray]], tuple[dict, jnp.ndarray]]:
    def loss_and_grad(params: ArrayTree, data_and_targets: tuple[jnp.ndarray, jnp.ndarray]):
        x_data, y_targets = data_and_targets
        
        predictions = jax.vmap(lambda x: network(params, x))(x_data)
        squared_errors = 1 - jnp.cos(jnp.imag(predictions - y_targets))
        loss_value = jnp.mean(squared_errors)
        
        # Compute gradients
        gradients = jax.grad(lambda p: jnp.mean(jnp.abs(jax.vmap(lambda x: network(p, x))(x_data) - y_targets)**2))(params)
        stats = {
            "loss": loss_value,
            "target": y_targets,
            "prediction": predictions
        }
        
        return stats, gradients
    
    return loss_and_grad