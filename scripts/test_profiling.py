#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.expandvars('$VIRTUAL_ENV/lib/python3.11/site-packages'))

import jax
import jax.numpy as jnp
from jax.profiler import trace, TraceAnnotation
import time

def test_profiling():
    """Simple test to verify JAX profiling works."""
    print("Testing JAX profiling...")
    
    # Create a simple computation
    def simple_computation():
        with TraceAnnotation("test_computation"):
            x = jnp.ones((1000, 1000))
            y = jnp.ones((1000, 1000))
            result = jnp.dot(x, y)
            return result
    
    # Create profiling directory
    profile_dir = f"../logs/test_profiling_{int(time.time())}"
    os.makedirs(profile_dir, exist_ok=True)
    
    print(f"Profiling directory: {profile_dir}")
    
    # Run with profiling
    with trace(profile_dir, create_perfetto_link=False, create_perfetto_trace=True):
        print("Starting profiler trace...")
        with TraceAnnotation("main_test"):
            result = simple_computation()
            print(f"Computation result shape: {result.shape}")
        print("Profiler trace completed.")
    
    print("Test completed successfully!")
    print(f"Check {profile_dir} for profiling data")

if __name__ == "__main__":
    test_profiling() 