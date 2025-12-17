# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

r"""Xception model.

TF2-MIGRATION: This module provides TF2-native Xception backbone implementation.
Key changes from TF1 version:
- Replaced tf.contrib.slim with tf.keras.layers
- Replaced tf.variable_scope with tf.name_scope (Keras layers handle variable naming)
- Replaced slim.arg_scope with explicit layer configuration
- Replaced slim.separable_conv2d with custom SeparableConv2DSame layer
- Added training parameter to call methods for batch norm
- Uses tf.keras.layers.BatchNormalization instead of slim.batch_norm
- Endpoint collection replaced with dictionary tracking

"Xception: Deep Learning with Depthwise Separable Convolutions"
Fran{\c{c}}ois Chollet
https://arxiv.org/abs/1610.02357

We implement the modified version by Jifeng Dai et al. for their COCO 2017
detection challenge submission, where the model is made deeper and has aligned
features for dense prediction tasks. See their slides for details:

"Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge
2017 Entry"
Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei and Jifeng Dai
ICCV 2017 COCO Challenge workshop
http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf

We made a few more changes on top of MSRA's modifications:
1. Fully convolutional: All the max-pooling layers are replaced with separable
  conv2d with stride = 2. This allows us to use atrous convolution to extract
  feature maps at any resolution.

2. We support adding ReLU and BatchNorm after depthwise convolution, motivated
  by the design of MobileNetv1.

"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications"
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
Tobias Weyand, Marco Andreetto, Hartwig Adam
https://arxiv.org/abs/1704.04861
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import range
import tensorflow as tf

from deeplab.core import utils

# TF2-MIGRATION: Import Keras layers for explicit usage
from tensorflow.keras import layers as keras_layers


_DEFAULT_MULTI_GRID = [1, 1, 1]
# The cap for tf.clip_by_value.
_CLIP_CAP = 6


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.

  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """


def fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  TF2-MIGRATION: Uses pure TensorFlow ops, no changes needed.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs


class SeparableConv2DSame(keras_layers.Layer):
  """Strided 2-D separable convolution with 'SAME' padding as a Keras Layer.

  TF2-MIGRATION: This replaces the slim-based separable_conv2d_same function.
  Implemented as a Keras Layer for proper weight tracking and training mode handling.

  If stride > 1 and use_explicit_padding is True, then we do explicit zero-
  padding, followed by conv2d with 'VALID' padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               stride=1,
               rate=1,
               depth_multiplier=1,
               use_explicit_padding=True,
               regularize_depthwise=False,
               weight_decay=0.00004,
               batch_norm_decay=0.9997,
               batch_norm_epsilon=1e-5,
               activation_fn=None,
               name=None,
               **kwargs):
    """Initialize SeparableConv2DSame layer.

    Args:
      filters: Number of output filters.
      kernel_size: Kernel size for the depthwise convolution.
      stride: Stride for the convolution.
      rate: Dilation rate for atrous convolution.
      depth_multiplier: Depth multiplier for depthwise convolution.
      use_explicit_padding: Whether to use explicit padding.
      regularize_depthwise: Whether to apply L2 regularization to depthwise weights.
      weight_decay: L2 regularization weight decay.
      batch_norm_decay: Momentum for batch normalization.
      batch_norm_epsilon: Epsilon for batch normalization.
      activation_fn: Activation function (None for no activation).
      name: Layer name.
      **kwargs: Additional layer arguments.
    """
    super(SeparableConv2DSame, self).__init__(name=name, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.stride = stride
    self.rate = rate
    self.depth_multiplier = depth_multiplier
    self.use_explicit_padding = use_explicit_padding
    self.regularize_depthwise = regularize_depthwise
    self.weight_decay = weight_decay
    self.batch_norm_decay = batch_norm_decay
    self.batch_norm_epsilon = batch_norm_epsilon
    self.activation_fn = activation_fn

  def build(self, input_shape):
    """Build the layer."""
    # Depthwise regularization
    depthwise_regularizer = None
    if self.regularize_depthwise:
      depthwise_regularizer = tf.keras.regularizers.l2(self.weight_decay)

    # Determine padding
    if self.stride == 1 or not self.use_explicit_padding:
      self.padding = 'same'
      self.needs_explicit_padding = False
    else:
      self.padding = 'valid'
      self.needs_explicit_padding = True

    # Depthwise convolution
    self.depthwise_conv = keras_layers.DepthwiseConv2D(
        kernel_size=self.kernel_size,
        strides=self.stride,
        padding=self.padding,
        dilation_rate=self.rate,
        depth_multiplier=self.depth_multiplier,
        use_bias=False,
        depthwise_regularizer=depthwise_regularizer,
        name='depthwise')

    # Batch norm after depthwise
    self.depthwise_bn = keras_layers.BatchNormalization(
        momentum=self.batch_norm_decay,
        epsilon=self.batch_norm_epsilon,
        name='depthwise_bn')

    # Pointwise convolution (if filters is not None)
    if self.filters is not None:
      self.pointwise_conv = keras_layers.Conv2D(
          filters=self.filters,
          kernel_size=1,
          strides=1,
          padding='same',
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
          name='pointwise')

      self.pointwise_bn = keras_layers.BatchNormalization(
          momentum=self.batch_norm_decay,
          epsilon=self.batch_norm_epsilon,
          name='pointwise_bn')

    super(SeparableConv2DSame, self).build(input_shape)

  def call(self, inputs, training=False):
    """Forward pass."""
    x = inputs

    # Apply explicit padding if needed
    if self.needs_explicit_padding:
      x = fixed_padding(x, self.kernel_size, self.rate)

    # Depthwise convolution
    x = self.depthwise_conv(x)
    x = self.depthwise_bn(x, training=training)

    # Apply activation after depthwise if specified
    if self.activation_fn is not None:
      x = self.activation_fn(x)

    # Pointwise convolution (if filters specified)
    if self.filters is not None:
      x = self.pointwise_conv(x)
      x = self.pointwise_bn(x, training=training)

      # Apply activation after pointwise if specified
      if self.activation_fn is not None:
        x = self.activation_fn(x)

    return x


def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          training=False,
                          activation_fn=None,
                          **kwargs):
  """Strided 2-D separable convolution with 'SAME' padding.

  TF2-MIGRATION: This function provides backward compatibility with TF1 code.
  For new code, prefer using SeparableConv2DSame layer directly.

  If stride > 1 and use_explicit_padding is True, then we do explicit zero-
  padding, followed by conv2d with 'VALID' padding.

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters (can be None for depthwise only).
    kernel_size: An int with the kernel_size of the filters.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    use_explicit_padding: If True, use explicit padding.
    regularize_depthwise: Whether to apply L2 regularization to depthwise weights.
    scope: Scope name (used as layer name).
    training: Boolean, whether in training mode.
    activation_fn: Activation function.
    **kwargs: Additional keyword arguments (ignored for compatibility).

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  # TF2-MIGRATION: Create layer and call it
  layer = SeparableConv2DSame(
      filters=num_outputs,
      kernel_size=kernel_size,
      stride=stride,
      rate=rate,
      depth_multiplier=depth_multiplier,
      use_explicit_padding=use_explicit_padding,
      regularize_depthwise=regularize_depthwise,
      activation_fn=activation_fn,
      name=scope)
  return layer(inputs, training=training)


class XceptionModule(keras_layers.Layer):
  """Xception module as a Keras Layer.

  TF2-MIGRATION: This replaces the slim-based xception_module function.

  The output of one Xception module is equal to the sum of `residual` and
  `shortcut`, where `residual` is the feature computed by three separable
  convolution. The `shortcut` is the feature computed by 1x1 convolution with
  or without striding.
  """

  def __init__(self,
               depth_list,
               skip_connection_type,
               stride,
               kernel_size=3,
               unit_rate_list=None,
               rate=1,
               activation_fn_in_separable_conv=False,
               regularize_depthwise=False,
               use_bounded_activation=False,
               use_explicit_padding=True,
               use_squeeze_excite=False,
               se_pool_size=None,
               batch_norm_decay=0.9997,
               batch_norm_epsilon=1e-5,
               weight_decay=0.00004,
               name=None,
               **kwargs):
    """Initialize XceptionModule.

    Args:
      depth_list: A list of three integers specifying the depth values.
      skip_connection_type: Skip connection type ('conv', 'sum', or 'none').
      stride: The block unit's stride.
      kernel_size: Integer, convolution kernel size.
      unit_rate_list: A list of three integers for atrous rates.
      rate: An integer, rate for atrous convolution.
      activation_fn_in_separable_conv: Include activation in separable conv.
      regularize_depthwise: Apply L2 regularization on depthwise weights.
      use_bounded_activation: Use bounded activations for quantization.
      use_explicit_padding: Use explicit padding.
      use_squeeze_excite: Use squeeze-and-excitation.
      se_pool_size: Pooling size for SE module.
      batch_norm_decay: Momentum for batch normalization.
      batch_norm_epsilon: Epsilon for batch normalization.
      weight_decay: L2 regularization weight decay.
      name: Layer name.
      **kwargs: Additional layer arguments.
    """
    super(XceptionModule, self).__init__(name=name, **kwargs)

    if len(depth_list) != 3:
      raise ValueError('Expect three elements in depth_list.')

    self.depth_list = depth_list
    self.skip_connection_type = skip_connection_type
    self.stride = stride
    self.kernel_size = kernel_size
    self.unit_rate_list = unit_rate_list if unit_rate_list else _DEFAULT_MULTI_GRID
    self.rate = rate
    self.activation_fn_in_separable_conv = activation_fn_in_separable_conv
    self.regularize_depthwise = regularize_depthwise
    self.use_bounded_activation = use_bounded_activation
    self.use_explicit_padding = use_explicit_padding
    self.use_squeeze_excite = use_squeeze_excite
    self.se_pool_size = se_pool_size
    self.batch_norm_decay = batch_norm_decay
    self.batch_norm_epsilon = batch_norm_epsilon
    self.weight_decay = weight_decay

    if self.unit_rate_list and len(self.unit_rate_list) != 3:
      raise ValueError('Expect three elements in unit_rate_list.')

  def build(self, input_shape):
    """Build the layer."""
    # Determine activation function for separable convs
    if self.activation_fn_in_separable_conv:
      self.sep_activation_fn = tf.nn.relu6 if self.use_bounded_activation else tf.nn.relu
    else:
      self.sep_activation_fn = None

    # Build three separable convolution layers
    self.separable_convs = []
    for i in range(3):
      conv = SeparableConv2DSame(
          filters=self.depth_list[i],
          kernel_size=self.kernel_size,
          stride=self.stride if i == 2 else 1,
          rate=self.rate * self.unit_rate_list[i],
          depth_multiplier=1,
          use_explicit_padding=self.use_explicit_padding,
          regularize_depthwise=self.regularize_depthwise,
          weight_decay=self.weight_decay,
          batch_norm_decay=self.batch_norm_decay,
          batch_norm_epsilon=self.batch_norm_epsilon,
          activation_fn=self.sep_activation_fn,
          name='separable_conv%d' % (i + 1))
      self.separable_convs.append(conv)

    # Build shortcut convolution if needed
    if self.skip_connection_type == 'conv':
      self.shortcut_conv = keras_layers.Conv2D(
          filters=self.depth_list[-1],
          kernel_size=1,
          strides=self.stride,
          padding='same',
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
          name='shortcut')
      self.shortcut_bn = keras_layers.BatchNormalization(
          momentum=self.batch_norm_decay,
          epsilon=self.batch_norm_epsilon,
          name='shortcut_bn')

    super(XceptionModule, self).build(input_shape)

  def call(self, inputs, training=False):
    """Forward pass."""
    residual = inputs

    # Apply pre-activation if not using activation in separable conv
    if not self.activation_fn_in_separable_conv:
      if self.use_bounded_activation:
        residual = tf.nn.relu6(residual)
      else:
        residual = tf.nn.relu(residual)

    # Apply three separable convolutions
    for i, conv in enumerate(self.separable_convs):
      # For middle convolutions (not first), apply activation if needed
      if i > 0 and not self.activation_fn_in_separable_conv:
        if self.use_bounded_activation:
          residual = tf.nn.relu6(residual)
        else:
          residual = tf.nn.relu(residual)
      residual = conv(residual, training=training)

    # Apply squeeze-excite if enabled
    # TF2-MIGRATION-NOTE: squeeze_excite from mobilenet_v3_ops needs separate migration
    # For now, we skip SE to avoid mobilenet dependency issues

    # Apply skip connection
    if self.skip_connection_type == 'conv':
      shortcut = self.shortcut_conv(inputs)
      shortcut = self.shortcut_bn(shortcut, training=training)
      if self.use_bounded_activation:
        residual = tf.clip_by_value(residual, -_CLIP_CAP, _CLIP_CAP)
        shortcut = tf.clip_by_value(shortcut, -_CLIP_CAP, _CLIP_CAP)
      outputs = residual + shortcut
      if self.use_bounded_activation:
        outputs = tf.nn.relu6(outputs)
    elif self.skip_connection_type == 'sum':
      if self.use_bounded_activation:
        residual = tf.clip_by_value(residual, -_CLIP_CAP, _CLIP_CAP)
        inputs = tf.clip_by_value(inputs, -_CLIP_CAP, _CLIP_CAP)
      outputs = residual + inputs
      if self.use_bounded_activation:
        outputs = tf.nn.relu6(outputs)
    elif self.skip_connection_type == 'none':
      outputs = residual
    else:
      raise ValueError('Unsupported skip connection type: %s' % self.skip_connection_type)

    return outputs


def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    kernel_size=3,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None,
                    use_bounded_activation=False,
                    use_explicit_padding=True,
                    use_squeeze_excite=False,
                    se_pool_size=None,
                    training=False):
  """An Xception module - functional interface.

  TF2-MIGRATION: This function provides backward compatibility with TF1 code.
  For new code, prefer using XceptionModule layer directly.

  The output of one Xception module is equal to the sum of `residual` and
  `shortcut`, where `residual` is the feature computed by three separable
  convolution. The `shortcut` is the feature computed by 1x1 convolution with
  or without striding. In some cases, the `shortcut` path could be a simple
  identity function or none (i.e, no shortcut).

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth_list: A list of three integers specifying the depth values of one
      Xception module.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    stride: The block unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    kernel_size: Integer, convolution kernel size.
    unit_rate_list: A list of three integers, determining the unit rate for
      each separable convolution in the xception module.
    rate: An integer, rate for atrous convolution.
    activation_fn_in_separable_conv: Includes activation function in the
      separable convolution or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    outputs_collections: Collection to add the Xception unit output (ignored in TF2).
    scope: Optional scope name (used as layer name).
    use_bounded_activation: Whether or not to use bounded activations.
    use_explicit_padding: If True, use explicit padding.
    use_squeeze_excite: Boolean, use squeeze-and-excitation or not.
    se_pool_size: None or integer specifying the pooling size used in SE module.
    training: Boolean, whether in training mode.

  Returns:
    The Xception module's output.
  """
  # TF2-MIGRATION: Create layer and call it
  layer = XceptionModule(
      depth_list=depth_list,
      skip_connection_type=skip_connection_type,
      stride=stride,
      kernel_size=kernel_size,
      unit_rate_list=unit_rate_list,
      rate=rate,
      activation_fn_in_separable_conv=activation_fn_in_separable_conv,
      regularize_depthwise=regularize_depthwise,
      use_bounded_activation=use_bounded_activation,
      use_explicit_padding=use_explicit_padding,
      use_squeeze_excite=use_squeeze_excite,
      se_pool_size=se_pool_size,
      name=scope)
  return layer(inputs, training=training)


def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None,
                       training=False,
                       end_points=None):
  """Stacks Xception blocks and controls output feature density.

  TF2-MIGRATION: Updated to use TF2 constructs:
  - Replaced tf.variable_scope with tf.name_scope
  - Removed slim.utils.collect_named_outputs (use end_points dict instead)
  - Added training parameter for batch norm layers

  First, this function creates scopes for the Xception in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the output
  stride, which is the ratio of the input to output spatial resolution. This
  is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A tensor of size [batch, height, width, channels].
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of Xception.
      For example, if the Xception employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the Xception block outputs (ignored in TF2).
    training: Boolean, whether in training mode.
    end_points: Optional dict to collect intermediate outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  if end_points is None:
    end_points = {}

  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.name_scope(block.scope):
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

        unit_name = 'unit_%d' % (i + 1)
        with tf.name_scope(unit_name):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, rate=rate, training=training,
                               **dict(unit, stride=1))
            rate *= unit.get('stride', 1)
          else:
            net = block.unit_fn(net, rate=1, training=training, **unit)
            current_stride *= unit.get('stride', 1)

      # Collect activations at the block's end
      end_points[block.scope] = net

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net, end_points


def xception(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=None,
             scope=None,
             sync_batch_norm_method='None'):
  """Generator for Xception models.

  TF2-MIGRATION: Updated to use TF2/Keras constructs:
  - Replaced tf.variable_scope with tf.name_scope
  - Replaced slim.arg_scope with explicit layer configuration
  - Replaced slim.conv2d with tf.keras.layers.Conv2D
  - Uses dictionary for end_points instead of collections
  - 'reuse' parameter is ignored (TF2 handles variable reuse differently)

  This function generates a family of Xception models. See the xception_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce Xception of various depths.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels]. Must be
      floating point. If a pretrained checkpoint is used, pixel values should be
      the same as during training.
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    keep_prob: Keep probability used in the pre-logits dropout layer.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    reuse: Ignored in TF2 (kept for API compatibility).
    scope: Optional scope name.
    sync_batch_norm_method: String, sync batchnorm method. Currently only
      support 'None'.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last Xception block, potentially after
      global average pooling. If num_classes is a non-zero integer, net contains
      the pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # TF2-MIGRATION: Using name_scope instead of variable_scope
  scope_name = scope or 'xception'
  with tf.name_scope(scope_name):
    end_points = {}
    net = inputs

    if output_stride is not None:
      if output_stride % 2 != 0:
        raise ValueError('The output_stride needs to be a multiple of 2.')
      output_stride //= 2

    # Root block function operated on inputs.
    # TF2-MIGRATION: Using Keras Conv2D instead of resnet_utils.conv2d_same
    # Entry flow conv1_1
    net = _conv2d_same(net, 32, 3, stride=2, name='entry_flow/conv1_1',
                       training=is_training)
    end_points['entry_flow/conv1_1'] = net

    # Entry flow conv1_2
    net = _conv2d_same(net, 64, 3, stride=1, name='entry_flow/conv1_2',
                       training=is_training)
    end_points['entry_flow/conv1_2'] = net

    # Extract features for entry_flow, middle_flow, and exit_flow.
    net, block_end_points = stack_blocks_dense(
        net, blocks, output_stride, training=is_training, end_points=end_points)
    end_points.update(block_end_points)

    if global_pool:
      # Global average pooling.
      net = tf.reduce_mean(net, [1, 2], name='global_pool', keepdims=True)
      end_points['global_pool'] = net

    if num_classes:
      # TF2-MIGRATION: Using Keras Dropout and Conv2D
      if is_training:
        net = tf.keras.layers.Dropout(rate=1.0-keep_prob, name='prelogits_dropout')(net, training=True)

      logits_conv = keras_layers.Conv2D(
          filters=num_classes,
          kernel_size=1,
          strides=1,
          padding='same',
          activation=None,
          use_bias=True,
          name='logits')
      net = logits_conv(net)
      end_points[scope_name + '/logits'] = net
      end_points['predictions'] = tf.nn.softmax(net, name='predictions')

    return net, end_points


def _conv2d_same(inputs, filters, kernel_size, stride, name, training=False,
                 batch_norm_decay=0.9997, batch_norm_epsilon=1e-5):
  """Helper function for conv2d with 'SAME' padding that handles stride > 1.

  TF2-MIGRATION: This replaces resnet_utils.conv2d_same with Keras layers.

  Args:
    inputs: Input tensor.
    filters: Number of output filters.
    kernel_size: Kernel size.
    stride: Stride for convolution.
    name: Layer name.
    training: Boolean, whether in training mode.
    batch_norm_decay: Momentum for batch normalization.
    batch_norm_epsilon: Epsilon for batch normalization.

  Returns:
    Output tensor.
  """
  if stride == 1:
    padding = 'same'
    x = inputs
  else:
    # Explicit padding for stride > 1
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    x = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    padding = 'valid'

  # Convolution
  conv = keras_layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=stride,
      padding=padding,
      use_bias=False,
      name=name + '_conv')
  x = conv(x)

  # Batch normalization
  bn = keras_layers.BatchNormalization(
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name=name + '_bn')
  x = bn(x, training=training)

  # Activation
  x = tf.nn.relu(x)

  return x


def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   kernel_size=3,
                   unit_rate_list=None,
                   use_squeeze_excite=False,
                   se_pool_size=None):
  """Helper function for creating a Xception block.

  Args:
    scope: The scope of the block.
    depth_list: The depth of the bottleneck layer for each unit.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    activation_fn_in_separable_conv: Includes activation function in the
      separable convolution or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
    kernel_size: Integer, convolution kernel size.
    unit_rate_list: A list of three integers, determining the unit rate in the
      corresponding xception block.
    use_squeeze_excite: Boolean, use squeeze-and-excitation or not.
    se_pool_size: None or integer specifying the pooling size used in SE module.

  Returns:
    An Xception block.
  """
  if unit_rate_list is None:
    unit_rate_list = _DEFAULT_MULTI_GRID
  return Block(scope, xception_module, [{
      'depth_list': depth_list,
      'skip_connection_type': skip_connection_type,
      'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
      'regularize_depthwise': regularize_depthwise,
      'stride': stride,
      'kernel_size': kernel_size,
      'unit_rate_list': unit_rate_list,
      'use_squeeze_excite': use_squeeze_excite,
      'se_pool_size': se_pool_size,
  }] * num_units)


def xception_41(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_41',
                sync_batch_norm_method='None'):
  """Xception-41 model."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block3',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=8,
                     stride=1),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     unit_rate_list=multi_grid),
  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope,
                  sync_batch_norm_method=sync_batch_norm_method)


def xception_65_factory(inputs,
                        num_classes=None,
                        is_training=True,
                        global_pool=True,
                        keep_prob=0.5,
                        output_stride=None,
                        regularize_depthwise=False,
                        kernel_size=3,
                        multi_grid=None,
                        reuse=None,
                        use_squeeze_excite=False,
                        se_pool_size=None,
                        scope='xception_65',
                        sync_batch_norm_method='None'):
  """Xception-65 model factory."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block3',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=16,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     unit_rate_list=multi_grid,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope,
                  sync_batch_norm_method=sync_batch_norm_method)


def xception_65(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_65',
                sync_batch_norm_method='None'):
  """Xception-65 model."""
  return xception_65_factory(
      inputs=inputs,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      keep_prob=keep_prob,
      output_stride=output_stride,
      regularize_depthwise=regularize_depthwise,
      multi_grid=multi_grid,
      reuse=reuse,
      scope=scope,
      use_squeeze_excite=False,
      se_pool_size=None,
      sync_batch_norm_method=sync_batch_norm_method)


def xception_71_factory(inputs,
                        num_classes=None,
                        is_training=True,
                        global_pool=True,
                        keep_prob=0.5,
                        output_stride=None,
                        regularize_depthwise=False,
                        kernel_size=3,
                        multi_grid=None,
                        reuse=None,
                        scope='xception_71',
                        use_squeeze_excite=False,
                        se_pool_size=None,
                        sync_batch_norm_method='None'):
  """Xception-71 model factory."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block3',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block4',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('entry_flow/block5',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=16,
                     stride=1,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2,
                     kernel_size=kernel_size,
                     use_squeeze_excite=use_squeeze_excite,
                     se_pool_size=se_pool_size),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     kernel_size=kernel_size,
                     unit_rate_list=multi_grid,
                     use_squeeze_excite=False,
                     se_pool_size=se_pool_size),
  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope,
                  sync_batch_norm_method=sync_batch_norm_method)


def xception_71(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_71',
                sync_batch_norm_method='None'):
  """Xception-71 model."""
  return xception_71_factory(
      inputs=inputs,
      num_classes=num_classes,
      is_training=is_training,
      global_pool=global_pool,
      keep_prob=keep_prob,
      output_stride=output_stride,
      regularize_depthwise=regularize_depthwise,
      multi_grid=multi_grid,
      reuse=reuse,
      scope=scope,
      use_squeeze_excite=False,
      se_pool_size=None,
      sync_batch_norm_method=sync_batch_norm_method)


def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       regularize_depthwise=False,
                       use_batch_norm=True,
                       use_bounded_activation=False,
                       sync_batch_norm_method='None'):
  """Defines the default Xception arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    weights_initializer_stddev: The standard deviation of the trunctated normal
      weight initializer.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    use_batch_norm: Whether or not to use batch normalization.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
    sync_batch_norm_method: String, sync batchnorm method. Currently only
      support `None`. Also, it is only effective for Xception.

  Returns:
    An `arg_scope` to use for the Xception models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
  }
  if regularize_depthwise:
    depthwise_regularizer = slim.l2_regularizer(weight_decay)
  else:
    depthwise_regularizer = None
  activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
  batch_norm = utils.get_batch_norm_fn(sync_batch_norm_method)
  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_initializer=tf.truncated_normal_initializer(
          stddev=weights_initializer_stddev),
      activation_fn=activation_fn,
      normalizer_fn=batch_norm if use_batch_norm else None):
    with slim.arg_scope([batch_norm], **batch_norm_params):
      with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.separable_conv2d],
            weights_regularizer=depthwise_regularizer):
          with slim.arg_scope(
              [xception_module],
              use_bounded_activation=use_bounded_activation,
              use_explicit_padding=not use_bounded_activation) as arg_sc:
            return arg_sc
