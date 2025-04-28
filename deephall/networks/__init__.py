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

from deephall.config import Network, NetworkType, System
from deephall.networks.laughlin import Laughlin
from deephall.networks.psiformer import Psiformer
from deephall.networks.parton import Parton


def make_network(system: System, network: Network) -> nn.Module:
    Q = system.flux / 2
    if network.type == NetworkType.laughlin:
        return Laughlin(
            flux=system.flux, nspins=system.nspins, excitation_lz=system.lz_center
        )
    if network.type == NetworkType.parton:
        return Parton(
            Q=Q,
            nspins=system.nspins,
            ndets=network.parton.determinants,
            num_heads=network.parton.num_heads,
            num_layers=network.parton.num_layers,
            heads_dim=network.parton.heads_dim,
            fermionic_type=network.parton.fermionic_type,
            flux_type=network.parton.flux_type,
            orbital_type=network.orbital,
        )
    
    if network.type == NetworkType.psiformer:
        return Psiformer(
            Q=Q,
            nspins=system.nspins,
            ndets=network.psiformer.determinants,
            num_heads=network.psiformer.num_heads,
            num_layers=network.psiformer.num_layers,
            heads_dim=network.psiformer.heads_dim,
            orbital_type=network.orbital,
        )
