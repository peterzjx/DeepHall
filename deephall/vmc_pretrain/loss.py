import jax
import jax.numpy as jnp
from typing import Callable
from deephall.types import ArrayTree, LogPsiNetwork

def make_vmc_fit_loss_fn(
    network: LogPsiNetwork ,
    loss_type: str = "phase"
) -> Callable[[ArrayTree, tuple[jnp.ndarray, jnp.ndarray]], tuple[dict, jnp.ndarray]]:
    def loss_and_grad(params: ArrayTree, data_and_targets: tuple[jnp.ndarray, jnp.ndarray]):
        x_data, y_targets = data_and_targets
        
        predictions = jax.vmap(lambda x: network(params, x))(x_data)
        # squared_errors = 1 - jnp.cos(jnp.imag(predictions - y_targets))
        if loss_type == "modulus":
            squared_errors = jnp.real(predictions - y_targets) ** 2
        elif loss_type == "phase":
            squared_errors = jnp.sin(jnp.imag(predictions - y_targets) - jnp.pi / 2) ** 2
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