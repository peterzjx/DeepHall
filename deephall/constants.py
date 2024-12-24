# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Ltd. and/or its affiliates
# ("Bytedance's Modifications"). All Bytedance's Modifications are
# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates.

import functools
from typing import TypeVar

import jax
from chex import ArrayTree
from jax import numpy as jnp

T = TypeVar("T", bound=jnp.ndarray | ArrayTree)

# Axis name we pmap over.
PMAP_AXIS_NAME = "qmc_pmap_axis"


# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
def pmap(func, *args, **kwargs):
    return functools.wraps(func)(
        jax.pmap(func, *args, **kwargs, axis_name=PMAP_AXIS_NAME)
    )


def pmean(x: T) -> T:
    return jax.lax.pmean(x, axis_name=PMAP_AXIS_NAME)
