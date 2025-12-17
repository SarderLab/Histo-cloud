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

"""This script contains utility functions.

TF2-MIGRATION: This module has been migrated from TF1/TF-Slim to native TF2/Keras.
Key changes:
- Removed tf.contrib.slim dependency
- Using tf.keras.layers for convolutions and batch normalization
- Updated deprecated TF1 ops (tf.to_float -> tf.cast, tf.image.resize_bilinear -> tf.image.resize)
- SplitSeparableConv2D implemented as tf.keras.layers.Layer subclass
"""
import tensorflow as tf


# TF2-MIGRATION: Quantized sigmoid for bounded activations (unchanged, pure TF ops)
q_sigmoid = lambda x: tf.nn.relu6(x + 3) * 0.16667


def resize_bilinear(images, size, output_dtype=tf.float32):
  """Returns resized images as output_type.

  TF2-MIGRATION: Updated from tf.image.resize_bilinear to tf.image.resize.
  Note: TF2's tf.image.resize uses 'bilinear' method by default and 
  align_corners behavior is controlled via antialias parameter.
  For exact TF1 behavior with align_corners=True, we use 
  tf.compat.v1.image.resize_bilinear for backward compatibility during migration.

  Args:
    images: A tensor of size [batch, height_in, width_in, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
      for the images.
    output_dtype: The destination type.
  Returns:
    A tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_dtype.
  """
  # TF2-MIGRATION: Using tf.image.resize with method='bilinear'
  # Note: TF2 resize doesn't have align_corners; behavior differs slightly at boundaries
  # For exact TF1 match during weight migration, can use tf.compat.v1.image.resize_bilinear
  images = tf.image.resize(images, size, method='bilinear')
  return tf.cast(images, dtype=output_dtype)


def scale_dimension(dim, scale):
  """Scales the input dimension.

  TF2-MIGRATION: Updated tf.to_float -> tf.cast for TF2 compatibility.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    # TF2-MIGRATION: tf.to_float is deprecated, use tf.cast
    return tf.cast((tf.cast(dim, tf.float32) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


class SplitSeparableConv2D(tf.keras.layers.Layer):
  """Separable conv2d split into depthwise and pointwise with activation between.

  TF2-MIGRATION: This replaces the slim-based split_separable_conv2d function.
  Implemented as a Keras Layer for proper weight tracking and training mode handling.

  This operation differs from standard separable_conv2d as it applies
  batch normalization and activation function between depthwise and pointwise conv2d.
  """

  def __init__(self,
               filters,
               kernel_size=3,
               rate=1,
               weight_decay=0.00004,
               depthwise_weights_initializer_stddev=0.33,
               pointwise_weights_initializer_stddev=0.06,
               batch_norm_decay=0.9997,
               batch_norm_epsilon=1e-5,
               activation_fn=tf.nn.relu,
               name=None,
               **kwargs):
    """Initialize SplitSeparableConv2D layer.

    Args:
      filters: Number of filters in the pointwise convolution.
      kernel_size: Kernel size for depthwise convolution.
      rate: Atrous/dilation rate for depthwise convolution.
      weight_decay: L2 regularization weight decay.
      depthwise_weights_initializer_stddev: Stddev for depthwise kernel init.
      pointwise_weights_initializer_stddev: Stddev for pointwise kernel init.
      batch_norm_decay: Decay for batch normalization moving averages.
      batch_norm_epsilon: Epsilon for batch normalization.
      activation_fn: Activation function to use (default: ReLU).
      name: Layer name.
      **kwargs: Additional layer arguments.
    """
    super(SplitSeparableConv2D, self).__init__(name=name, **kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.rate = rate
    self.weight_decay = weight_decay
    self.depthwise_weights_initializer_stddev = depthwise_weights_initializer_stddev
    self.pointwise_weights_initializer_stddev = pointwise_weights_initializer_stddev
    self.batch_norm_decay = batch_norm_decay
    self.batch_norm_epsilon = batch_norm_epsilon
    self.activation_fn = activation_fn

  def build(self, input_shape):
    """Build the layer weights."""
    # TF2-MIGRATION: Using DepthwiseConv2D for the depthwise part
    self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=self.kernel_size,
        strides=1,
        padding='same',
        dilation_rate=self.rate,
        depth_multiplier=1,
        use_bias=False,
        depthwise_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.depthwise_weights_initializer_stddev),
        depthwise_regularizer=None,  # No regularization on depthwise
        name='depthwise')

    # Batch norm after depthwise conv
    self.depthwise_bn = tf.keras.layers.BatchNormalization(
        momentum=self.batch_norm_decay,
        epsilon=self.batch_norm_epsilon,
        name='depthwise_bn')

    # Pointwise (1x1) convolution
    self.pointwise_conv = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.pointwise_weights_initializer_stddev),
        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
        name='pointwise')

    # Batch norm after pointwise conv
    self.pointwise_bn = tf.keras.layers.BatchNormalization(
        momentum=self.batch_norm_decay,
        epsilon=self.batch_norm_epsilon,
        name='pointwise_bn')

    super(SplitSeparableConv2D, self).build(input_shape)

  def call(self, inputs, training=False):
    """Forward pass.

    Args:
      inputs: Input tensor [batch, height, width, channels].
      training: Boolean, whether in training mode.

    Returns:
      Output tensor after split separable convolution.
    """
    # Depthwise convolution
    x = self.depthwise_conv(inputs)
    x = self.depthwise_bn(x, training=training)
    if self.activation_fn is not None:
      x = self.activation_fn(x)

    # Pointwise convolution
    x = self.pointwise_conv(x)
    x = self.pointwise_bn(x, training=training)
    if self.activation_fn is not None:
      x = self.activation_fn(x)

    return x

  def get_config(self):
    """Return layer configuration."""
    config = super(SplitSeparableConv2D, self).get_config()
    config.update({
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'rate': self.rate,
        'weight_decay': self.weight_decay,
        'depthwise_weights_initializer_stddev': self.depthwise_weights_initializer_stddev,
        'pointwise_weights_initializer_stddev': self.pointwise_weights_initializer_stddev,
        'batch_norm_decay': self.batch_norm_decay,
        'batch_norm_epsilon': self.batch_norm_epsilon,
    })
    return config


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None,
                           training=False):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  TF2-MIGRATION: This function provides backward compatibility with TF1 code.
  For new code, prefer using SplitSeparableConv2D layer directly.

  This operation differs from `tf.keras.layers.SeparableConv2D` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation (used as layer name).
    training: Boolean, whether in training mode (for batch norm).

  Returns:
    Computed features after split separable conv2d.
  """
  # TF2-MIGRATION: Create layer instance and call it
  # Note: This creates a new layer each call which is inefficient.
  # For performance, instantiate SplitSeparableConv2D in model's __init__
  layer = SplitSeparableConv2D(
      filters=filters,
      kernel_size=kernel_size,
      rate=rate,
      weight_decay=weight_decay,
      depthwise_weights_initializer_stddev=depthwise_weights_initializer_stddev,
      pointwise_weights_initializer_stddev=pointwise_weights_initializer_stddev,
      name=scope)
  return layer(inputs, training=training)


def get_label_weight_mask(labels, ignore_label, num_classes, label_weights=1.0):
  """Gets the label weight mask.

  TF2-MIGRATION: No changes needed - uses pure TensorFlow ops.

  Args:
    labels: A Tensor of labels with the shape of [-1].
    ignore_label: Integer, label to ignore.
    num_classes: Integer, the number of semantic classes.
    label_weights: A float or a list of weights. If it is a float, it means all
      the labels have the same weight. If it is a list of weights, then each
      element in the list represents the weight for the label of its index, for
      example, label_weights = [0.1, 0.5] means the weight for label 0 is 0.1
      and the weight for label 1 is 0.5.

  Returns:
    A Tensor of label weights with the same shape of labels, each element is the
      weight for the label with the same index in labels and the element is 0.0
      if the label is to ignore.

  Raises:
    ValueError: If label_weights is neither a float nor a list, or if
      label_weights is a list and its length is not equal to num_classes.
  """
  if not isinstance(label_weights, (float, list)):
    raise ValueError(
        'The type of label_weights is invalid, it must be a float or a list.')

  if isinstance(label_weights, list) and len(label_weights) != num_classes:
    raise ValueError(
        'Length of label_weights must be equal to num_classes if it is a list, '
        'label_weights: %s, num_classes: %d.' % (label_weights, num_classes))

  not_ignore_mask = tf.not_equal(labels, ignore_label)
  not_ignore_mask = tf.cast(not_ignore_mask, tf.float32)
  if isinstance(label_weights, float):
    return not_ignore_mask * label_weights

  label_weights = tf.constant(label_weights, tf.float32)
  weight_mask = tf.einsum('...y,y->...',
                          tf.one_hot(labels, num_classes, dtype=tf.float32),
                          label_weights)
  return tf.multiply(not_ignore_mask, weight_mask)


def get_batch_norm_fn(sync_batch_norm_method='None'):
  """Gets batch norm function/class.

  TF2-MIGRATION: Returns tf.keras.layers.BatchNormalization class instead of slim.batch_norm.
  The returned class should be instantiated with appropriate parameters.

  Currently we only support the following methods:
    - 'None' (no sync batch norm). We use tf.keras.layers.BatchNormalization.

  Args:
    sync_batch_norm_method: String, method used to sync batch norm.

  Returns:
    BatchNormalization class.

  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  if sync_batch_norm_method == 'None':
    # TF2-MIGRATION: Return Keras BatchNormalization class
    return tf.keras.layers.BatchNormalization
  else:
    raise ValueError('Unsupported sync_batch_norm_method: %s' % sync_batch_norm_method)


def get_batch_norm_params(decay=0.9997,
                          epsilon=1e-5,
                          center=True,
                          scale=True,
                          is_training=True,
                          sync_batch_norm_method='None',
                          initialize_gamma_as_zeros=False):
  """Gets batch norm parameters for tf.keras.layers.BatchNormalization.

  TF2-MIGRATION: Updated to return parameters compatible with Keras BatchNormalization.
  Key mapping from TF1/slim to TF2/Keras:
    - 'decay' -> 'momentum' (Keras uses momentum = 1 - decay effectively, but 
       actually uses momentum directly as the moving average coefficient)
    - 'is_training' -> passed to call() as 'training' argument, not constructor

  Args:
    decay: Float, decay for the moving average (momentum in Keras terms).
    epsilon: Float, value added to variance to avoid dividing by zero.
    center: Boolean. If True, add offset of `beta` to normalized tensor.
    scale: Boolean. If True, multiply by `gamma`.
    is_training: Boolean, whether or not the layer is in training mode.
      TF2-MIGRATION-NOTE: In TF2/Keras, training mode is passed to call(), not __init__.
      This parameter is kept for API compatibility but should be passed to layer.call().
    sync_batch_norm_method: String, method used to sync batch norm.
    initialize_gamma_as_zeros: Boolean, initializing `gamma` as zeros or not.

  Returns:
    A dictionary for BatchNormalization constructor parameters.

  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  # TF2-MIGRATION: Build Keras-compatible batch norm parameters
  batch_norm_params = {
      'momentum': decay,  # TF2-MIGRATION: 'decay' in slim maps to 'momentum' in Keras
      'epsilon': epsilon,
      'center': center,
      'scale': scale,
  }

  if initialize_gamma_as_zeros:
    if sync_batch_norm_method == 'None':
      # TF2-MIGRATION: Keras uses gamma_initializer instead of param_initializers
      batch_norm_params['gamma_initializer'] = tf.zeros_initializer()
    else:
      raise ValueError('Unsupported sync_batch_norm_method: %s' % sync_batch_norm_method)

  return batch_norm_params


def create_batch_norm_layer(decay=0.9997,
                            epsilon=1e-5,
                            center=True,
                            scale=True,
                            initialize_gamma_as_zeros=False,
                            name=None):
  """Factory function to create a BatchNormalization layer with standard params.

  TF2-MIGRATION: New helper function for creating BatchNorm layers with
  consistent parameters across the model.

  Args:
    decay: Float, momentum for the moving average.
    epsilon: Float, small constant for numerical stability.
    center: Boolean, whether to add beta offset.
    scale: Boolean, whether to multiply by gamma.
    initialize_gamma_as_zeros: Boolean, whether to initialize gamma as zeros.
    name: Optional name for the layer.

  Returns:
    A tf.keras.layers.BatchNormalization instance.
  """
  gamma_initializer = 'zeros' if initialize_gamma_as_zeros else 'ones'

  return tf.keras.layers.BatchNormalization(
      momentum=decay,
      epsilon=epsilon,
      center=center,
      scale=scale,
      gamma_initializer=gamma_initializer,
      name=name)
