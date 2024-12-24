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

from collections.abc import Mapping, Sequence
from typing import Any, cast

import jax
import kfac_jax
from chex import PRNGKey
from jax import numpy as jnp

from deephall import constants
from deephall.config import OptimizerKfac
from deephall.log import CheckpointState
from deephall.loss import LossStats
from deephall.types import TrainingInit, TrainingStep

repeated_dense_tag = kfac_jax.LayerTag("repeated_dense", num_inputs=1, num_outputs=1)


def register_repeated_dense(y, x, w, b=None):
    if b is None:
        return repeated_dense_tag.bind(y, x, w)
    return repeated_dense_tag.bind(y, x, w, b)


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    """Dense block that is repeatedly applied to multiple inputs (e.g. vmap)."""

    def parameters_shaped_list_to_array(
        self,
        parameters_shaped_list: Sequence[kfac_jax.utils.Array],
    ) -> kfac_jax.utils.Array:
        for p, s in zip(parameters_shaped_list, self.parameters_shapes):
            assert p.shape == s

        if self.has_bias:
            w, b = parameters_shaped_list
            return jnp.concatenate(
                [w.reshape([-1, b.size]), b.reshape([1, -1])], axis=0
            )

        else:
            [w] = parameters_shaped_list
            return w.reshape([w.shape[0], -1])

    def array_to_parameters_shaped_list(
        self, array: kfac_jax.utils.Array
    ) -> tuple[kfac_jax.utils.Array, ...]:
        if self.has_bias:
            w, b = array[:-1], array[-1]
            w_shape, b_shape = self.parameters_shapes
            return w.reshape(w_shape), b.reshape(b_shape)

        else:
            return tuple([array.reshape(self.parameters_shapes[0])])

    def fixed_scale(self) -> kfac_jax.utils.Numeric:
        (x_shape,) = self.inputs_shapes
        return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.KroneckerFactored.State,  # type: ignore
        estimation_data: Mapping[str, Sequence[kfac_jax.utils.Array]],
        ema_old: kfac_jax.utils.Numeric,
        ema_new: kfac_jax.utils.Numeric,
        batch_size: kfac_jax.utils.Numeric,
    ) -> kfac_jax.KroneckerFactored.State:
        [x] = estimation_data["inputs"]
        [dy] = estimation_data["outputs_tangent"]
        assert x.shape[0] == batch_size

        batch_size = x.size // (self.array_shape[0] - self.has_bias)
        estimation_data = {
            **estimation_data,
            "inputs": (x.real.reshape((batch_size, -1)),),
            "outputs_tangent": (dy.real.reshape((batch_size, -1)),),
        }

        return super().update_curvature_matrix_estimate(
            state=state,
            estimation_data=estimation_data,
            ema_old=ema_old,
            ema_new=ema_new,
            batch_size=batch_size,
        )


def _repeated_dense(
    x: kfac_jax.utils.Array, params: Sequence[kfac_jax.utils.Array]
) -> kfac_jax.utils.Array:
    """Example of a dense layer function."""
    w, *opt_b = params
    y = jax.lax.dot_general(x, w, (((x.ndim - 1,), (0,)), ((), ())))
    if opt_b:
        b = opt_b[0]
        y += b.reshape((1, *b.shape))
    return y


def _repeated_dense_attention_out(
    x: kfac_jax.utils.Array, params: Sequence[kfac_jax.utils.Array]
) -> kfac_jax.utils.Array:
    """Example of a dense layer function."""
    w, b = params
    y = jax.lax.dot_general(x, w, (((x.ndim - 2, x.ndim - 1), (0, 1)), ((), ())))
    y += b.reshape((1, *b.shape))
    return y


def _repeated_dense_complex_no_bias(
    x: kfac_jax.utils.Array, params: Sequence[kfac_jax.utils.Array]
) -> kfac_jax.utils.Array:
    [w] = params
    w = w.astype(jnp.complex64)
    y = jax.lax.dot_general(x, w, (((x.ndim - 1,), (0,)), ((), ())))
    return y


def _dense_parameter_extractor(
    eqns: Sequence[jax.core.JaxprEqn],
) -> Mapping[str, Any]:
    """Extracts all parameters from the conv_general_dilated operator."""
    for eqn in eqns:
        if eqn.primitive.name == "dot_general":
            return dict(**eqn.params)
    assert False


GRAPH_PATTERNS = (
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_with_bias",
        tag_primitive=repeated_dense_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([2, 3, 4]), [jnp.zeros([4, 3]), jnp.zeros([3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_no_bias",
        tag_primitive=repeated_dense_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([2, 3, 4]), [jnp.zeros([4, 3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_more_dim",
        tag_primitive=repeated_dense_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([1, 2, 3, 4]), [jnp.zeros([4, 3]), jnp.zeros(3)]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_more_dim_no_bias",
        tag_primitive=repeated_dense_tag,
        compute_func=_repeated_dense,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([1, 2, 3, 4]), [jnp.zeros([4, 3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_complex_no_bias",
        tag_primitive=repeated_dense_tag,
        compute_func=_repeated_dense_complex_no_bias,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([2, 3, 4], dtype=jnp.complex64), [jnp.zeros([4, 3])]],
    ),
    kfac_jax.tag_graph_matcher.GraphPattern(
        name="repeated_dense_attention_with_bias",
        tag_primitive=repeated_dense_tag,
        compute_func=_repeated_dense_attention_out,
        parameters_extractor_func=_dense_parameter_extractor,
        example_args=[jnp.zeros([1, 2, 3, 4]), [jnp.zeros([3, 4, 3]), jnp.zeros([3])]],
    ),
    *kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS,
)

kfac_jax.set_default_tag_to_block_ctor("repeated_dense", RepeatedDenseBlock)


def make_kfac_training_step(
    optim_cfg: OptimizerKfac, loss_grad_fn
) -> tuple[TrainingInit, TrainingStep]:
    def val_and_grad(params, data):
        stats, grads = loss_grad_fn(params, data)
        return (stats["energy"], stats), grads

    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=0.0,
        norm_constraint=1e-3,
        value_func_has_aux=True,
        learning_rate_schedule=optim_cfg.lr.schedule,
        curvature_ema=0.95,
        inverse_update_period=1,
        min_damping=1e-4,
        num_burnin_steps=0,
        register_only_generic=False,
        estimation_mode="fisher_exact",
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=GRAPH_PATTERNS,
        ),
    )
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(1e-3))

    def init(params, key, data):
        return optimizer.init(params, key, data)

    def step(state: CheckpointState, key: PRNGKey):
        params, data, opt_state, mcmc_width = state
        params, opt_state, *_, stats = optimizer.step(
            params=params,
            state=opt_state,
            rng=key,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )
        return (
            CheckpointState(params, data, opt_state, mcmc_width),
            cast(LossStats, stats["aux"]),
        )

    return init, step
