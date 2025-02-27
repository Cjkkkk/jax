# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax._src import core
from jax.interpreters import xla, mlir
from jax.interpreters.mlir import ir, hlo
import functools

def _cudnn_pw_abstract_eval(arg, *, pw_mode):
  return core.ShapedArray(arg.shape, arg.dtype)

def _cudnn_pw_bwd_abstract_eval(arg, grad, *, pw_mode):
  return core.ShapedArray(arg.shape, arg.dtype)

def _cudnn_pw_fwd_rule(arg, pw_mode):
  return cudnn_pw.bind(arg, pw_mode=pw_mode), (arg,)

def _cudnn_pw_bwd_rule(pw_mode, res, grad):
  (arg,) = res
  grad = cudnn_pw_bwd.bind(arg, grad, pw_mode=pw_mode)
  return (grad,)

def _cudnn_pw_lowering(
  ctx,
  arg,
  pw_mode
):
  arg_type = ir.RankedTensorType(arg.type)
  cudnn_pw = hlo.CustomCallOp(
    [arg_type],
    [arg],
    call_target_name="__cudnn$pw",
  )
  return cudnn_pw.results

def _cudnn_pw_bwd_lowering(
  ctx,
  arg,
  grad,
  pw_mode
):
  arg_type = ir.RankedTensorType(arg.type)
  cudnn_pw = hlo.CustomCallOp(
    [arg_type],
    [arg, grad],
    call_target_name="__cudnn$pw_bwd",
  )
  return cudnn_pw.results

cudnn_pw = core.Primitive("cudnn_pw")
cudnn_pw.multiple_results = False
cudnn_pw.def_impl(
  functools.partial(xla.apply_primitive, cudnn_pw))
cudnn_pw.def_abstract_eval(_cudnn_pw_abstract_eval)
mlir.register_lowering(
  cudnn_pw, _cudnn_pw_lowering, platform="cuda"
)

cudnn_pw_bwd = core.Primitive("cudnn_pw_bwd")
cudnn_pw_bwd.multiple_results = False
cudnn_pw_bwd.def_impl(
  functools.partial(xla.apply_primitive, cudnn_pw_bwd))
cudnn_pw_bwd.def_abstract_eval(_cudnn_pw_bwd_abstract_eval)
mlir.register_lowering(
  cudnn_pw_bwd, _cudnn_pw_bwd_lowering, platform="cuda"
)

@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _cudnn_pw(arg, pw_mode):
  return cudnn_pw.bind(arg, pw_mode=pw_mode)

_cudnn_pw.defvjp(
  _cudnn_pw_fwd_rule, _cudnn_pw_bwd_rule
)

def tanh(arg):
  return _cudnn_pw(arg, "tanh")

if __name__ == "__main__":
  import jax.numpy as jnp
  query = jnp.zeros((5,6), dtype=jnp.float32)
  grad = jnp.ones((5,6), dtype=jnp.float32)

  def train(query, grad):
    out, grad_fun = jax.vjp(tanh, query)
    query_grad = grad_fun(grad)
    return out, query_grad

  def get_hlo(func, *args):
    print(jax.jit(func).lower(*args).as_text("hlo"))
  get_hlo(train, query, grad)
