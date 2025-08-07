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

from flax import linen as nn

from deephall.config import VNetwork, NetworkType, System
from deephall.velocity_networks.super_laughlin_v import SuperLaughlinVelocity
from deephall.velocity_networks.laughlin_v import LaughlinVelocity


def make_v_network(system: System, network: VNetwork) -> nn.Module:
    Q = system.flux / 2
    if network.type == NetworkType.laughlin_v:
        return LaughlinVelocity(
            flux=system.flux, nspins=system.nspins
        )
    if network.type == NetworkType.super_laughlin_v:
        return SuperLaughlinVelocity(
            flux=system.flux, nspins=system.nspins
        )
