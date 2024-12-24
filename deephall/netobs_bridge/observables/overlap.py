# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Any

import jax
from jax import numpy as jnp
from netobs.observables import Estimator, Observable

from deephall.netobs_bridge.hall_system import HallSystem
from deephall.networks import make_network


class Overlap(Observable):
    def shapeof(self, system) -> tuple[int, ...]:
        return ()


class OverlapEstimator(Estimator[HallSystem]):
    observable_type = Overlap

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.batch_network = jax.pmap(
            jax.vmap(adaptor.call_network, in_axes=(None, 0, None)),
            in_axes=(0, 0, None),
        )
        cfg = adaptor.cfg
        laughlin = make_network(
            cfg.system, dataclasses.replace(cfg.network, type="laughlin")
        )
        self.batch_laughlin = jax.pmap(jax.vmap(laughlin.apply, in_axes=(None, 0)))

    def empty_val_state(
        self, steps: int
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        empty_values = {
            "ratio": jnp.zeros(steps, "complex64"),
            "ratio_square": jnp.zeros(steps),
        }
        return empty_values, {}

    def evaluate(
        self, i, params, key, data, system, state, aux_data
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del i, aux_data, key
        logpsi = self.batch_network(params, data, system)
        logphi = self.batch_laughlin(params, data)
        shift = jnp.mean(logphi - logpsi)
        ratio = jnp.exp(logphi - logpsi - shift)
        return {"ratio": ratio, "ratio_square": jnp.abs(ratio) ** 2}, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del state
        ratio = all_values["ratio"]
        ratio_square = all_values["ratio_square"]
        overlap = jnp.abs(jnp.nanmean(ratio)) ** 2 / jnp.nanmean(ratio_square)
        return {"overlap": overlap}


DEFAULT = OverlapEstimator  # Useful in CLI
