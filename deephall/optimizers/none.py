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

from chex import PRNGKey

from deephall import constants
from deephall.log import CheckpointState
from deephall.types import TrainingInit, TrainingStep


def make_inference_step(loss_grad_fn) -> tuple[TrainingInit, TrainingStep]:
    @constants.pmap
    def init(params, key, data):
        del params, key, data
        return None

    @constants.pmap
    def step(state: CheckpointState, key: PRNGKey):
        del key
        params, data, opt_state, mcmc_width = state
        stats, _ = loss_grad_fn(params, data)
        return (CheckpointState(params, data, opt_state, mcmc_width), stats)

    return init, step
