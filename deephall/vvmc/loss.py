import jax
import jax.numpy as jnp
from typing import Callable
from deephall.types import ArrayTree, LogPsiNetwork
from deephall.config import System

def make_vvmc_fit_loss_fn(
    network: LogPsiNetwork
) -> Callable[[ArrayTree, tuple[jnp.ndarray, jnp.ndarray]], tuple[dict, jnp.ndarray]]:
    def loss_and_grad(params: ArrayTree, data_and_targets: tuple[jnp.ndarray, jnp.ndarray]):
        x_data, y_targets = data_and_targets
        
        # Compute predictions for all x_data
        predictions = jax.vmap(lambda x: network(params, x))(x_data)
        
        # Compute MSE between predictions and targets
        squared_errors = jnp.abs(predictions - y_targets)**2
        loss_value = jnp.mean(squared_errors)
        
        # Compute gradients
        gradients = jax.grad(lambda p: jnp.mean(jnp.abs(jax.vmap(lambda x: network(p, x))(x_data) - y_targets)**2))(params)
        
        # Create stats dictionary
        stats = {
            "loss": loss_value,
            "target": y_targets,
            "prediction": predictions
        }
        
        return stats, gradients
    
    return loss_and_grad

