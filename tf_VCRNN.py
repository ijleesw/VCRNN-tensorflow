"""
Implementation of VCRNN and VCGRU based on a paper

"VARIABLE COMPUTATION IN RECURRENT NEURAL NETWORKS" (Jernite et al., 2017).

(https://openreview.net/pdf?id=S1LVSrcge)

"""

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


# @tf_export("nn.rnn_cell.VCRNNCell")
class VCRNNCell(_LayerRNNCell):
  """VCRNN cell (cf. https://openreview.net/pdf?id=S1LVSrcge).

  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    sharpness: $\lambda$ from the paper.
    epsilon: Used to compute Thres_{\epsilon} from the paper.
    m_target: Used to add L1 loss.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self, num_units, sharpness=0.1, epsilon = 0.5,
                  m_target=0.35, activation=None, reuse=None, name=None):
    super(VCRNNCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._sharpness = sharpness
    self._epsilon = epsilon
    self._m_target = m_target
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value

    # for computing m_t
    self._u = self.add_variable(
        "u",
        shape=[self._num_units, 1])
    self._v = self.add_variable(
        "v",
        shape=[input_depth, 1])
    self._b = self.add_variable(
        "b",
        shape=[1],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    # for Elman Gate
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    add = math_ops.add
    sub = math_ops.subtract
    mult = math_ops.multiply

    # computing m_t
    m_t = add(math_ops.matmul(state, self._u),
              math_ops.matmul(inputs, self._v))
    m_t = nn_ops.bias_add(m_t, self._b)
    m_t = math_ops.sigmoid(m_t)

    # add L1 loss
    ops.add_to_collection('L1 loss', math_ops.abs(m_t - self._m_target))

    # computing e_t (= thr)
    i = gen_math_ops._range(1, self._num_units+1, 1)
    i = math_ops.cast(i, dtype=dtypes.float32)
    mtD = gen_array_ops.tile(mult(m_t[1], self._num_units), [self._num_units])
    thr = math_ops.sigmoid(mult(self._sharpness, sub(mtD, i)))
    thr = math_ops.round(add(thr, sub(0.5, self._epsilon)))
    ones = array_ops.ones_like(thr)
    thr_inv = sub(ones, thr)
    
    # computing h_t
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    output = add(mult(gate_inputs, thr), mult(state, thr_inv))
    
    return output, output


# @tf_export("nn.rnn_cell.VCGRUCell")
class VCGRUCell(_LayerRNNCell):
  """VCGRU cell (cf. https://openreview.net/pdf?id=S1LVSrcge).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    sharpness: $\lambda$ from the paper.
    epsilon: Used to compute Thres_{\epsilon} from the paper.
    m_target: Used to add L1 loss.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               sharpness=0.1,
               epsilon=0.5,
               m_target=0.35,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(VCGRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._sharpness = sharpness
    self._epsilon = epsilon
    self._m_target = m_target
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value

    # for computing m_t
    self._u = self.add_variable(
        "u",
        shape=[self._num_units, 1])
    self._v = self.add_variable(
        "v",
        shape=[input_depth, 1])
    self._b = self.add_variable(
        "b",
        shape=[1],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    # for the else
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """VCGRU with nunits cells."""
    add = math_ops.add
    sub = math_ops.subtract
    mult = math_ops.multiply

    # computing m_t
    m_t = add(math_ops.matmul(state, self._u),
              math_ops.matmul(inputs, self._v))
    m_t = nn_ops.bias_add(m_t, self._b)
    m_t = math_ops.sigmoid(m_t)

    # add L1 loss
    ops.add_to_collection('L1 loss', math_ops.abs(m_t - self._m_target))

    # computing e_t (= thr)
    i = gen_math_ops._range(1, self._num_units+1, 1)
    i = math_ops.cast(i, dtype=dtypes.float32)
    mtD = gen_array_ops.tile(mult(m_t[1], self._num_units), [self._num_units])
    thr = math_ops.sigmoid(mult(self._sharpness, sub(mtD, i)))
    thr = math_ops.round(add(thr, sub(0.5, self._epsilon)))
    ones = array_ops.ones_like(thr)
    thr_inv = sub(ones, thr)

    # computing h_t
    if inputs.shape[1] < thr.shape[0]:
      _inputs = mult(inputs, array_ops.slice(thr, [0,], [inputs.shape[1],]))
    elif inputs.shape[1] > thr.shape[0]:
      _inputs = mult(inputs,
                     array_ops.concat(1, [thr, array_ops.zeros_like(inputs.shape[1] - thr.shape[0])]))
    else:
      _inputs = mult(inputs, thr)
    _state = mult(state, thr)

    gate_inputs = math_ops.matmul(
        array_ops.concat([_inputs, _state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    _r_state = r * _state

    candidate = math_ops.matmul(
        array_ops.concat([_inputs, _r_state], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * _state + (1 - u) * c

    output = add(mult(new_h, thr), mult(state, thr_inv))
    return output, output
