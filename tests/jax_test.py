import os

# Set environment variables BEFORE importing jax
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or to all GPUs: "0,1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

import jax

print("Outside test:", jax.devices())

def test_jax_cuda():
    print("Inside test:", jax.devices())
    assert any(d.platform == "gpu" for d in jax.devices()), "No GPU devices found"
