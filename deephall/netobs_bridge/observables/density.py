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

from typing import Any

from jax import numpy as jnp
from netobs.observables import Estimator
from netobs.observables.density import Density

from deephall.netobs_bridge.hall_system import HallSystem


class DensityEstimator(Estimator[HallSystem]):
    observable_type = Density

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.hist_bins = self.options.get("bins", 50)

    def empty_val_state(
        self, steps: int
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del steps
        return {}, {"map": jnp.zeros(self.hist_bins)}

    def evaluate(
        self, i, params, key, data, system, state, aux_data
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del i, params, system, aux_data, key
        theta = jnp.reshape(data[..., 0], (-1, 1))
        hist_range = [(0.0, jnp.pi)]
        state["map"] += jnp.histogramdd(theta, self.hist_bins, hist_range)[0]
        return {}, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del all_values, state
        return {}


DEFAULT = DensityEstimator  # Useful in CLI
