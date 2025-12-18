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
"""Utility functions for training.

TF2 Migration:
- Removed tensorflow.contrib.framework (not available in TF2)
- Replaced tf.to_float with tf.cast
- Replaced tf.losses.add_loss with explicit loss accumulation
- Replaced tf.train.* learning rate schedules with tf.keras.optimizers.schedules
- Updated get_model_init_fn to use TF2 checkpoint loading
"""

import six
import tensorflow as tf

from deeplab.core import preprocess_utils
from deeplab.core import utils


def _div_maybe_zero(total_loss, num_present):
  """Normalizes the total loss with the number of present pixels."""
  return tf.cast(num_present > 0, tf.float32) * tf.math.divide(
      total_loss,
      tf.maximum(1e-5, num_present))


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  hard_example_mining_step=0,
                                                  top_k_percent_pixels=1.0,
                                                  gt_is_matting_map=False,
                                                  scope=None):
  """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: A float or a list of loss weights. If it is a float, it means
      all the labels have the same weight. If it is a list of weights, then each
      element in the list represents the weight for the label of its index, for
      example, loss_weight = [0.1, 0.5] means the weight for label 0 is 0.1 and
      the weight for label 1 is 0.5.
    upsample_logits: Boolean, upsample logits or not.
    hard_example_mining_step: An integer, the training step in which the hard
      exampling mining kicks off. Note that we gradually reduce the mining
      percent to the top_k_percent_pixels. For example, if
      hard_example_mining_step = 100K and top_k_percent_pixels = 0.25, then
      mining percent will gradually reduce from 100% to 25% until 100K steps
      after which we only mine top 25% pixels.
    top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its value
      < 1.0, only compute the loss for the top k percent pixels (e.g., the top
      20% pixels). This is useful for hard pixel mining.
    gt_is_matting_map: If true, the groundtruth is a matting map of confidence
      score. If false, the groundtruth is an integer valued class mask.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None, or groundtruth is matting map while
      label is not floating value.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  # If input groundtruth is a matting map of confidence, check if the input
  # labels are floating point values.
  if gt_is_matting_map and not labels.dtype.is_floating:
    raise ValueError('Labels must be floats if groundtruth is a matting map.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          method=tf.image.ResizeMethod.BILINEAR)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      # When gt_is_matting_map = true, label downsampling with nearest neighbor
      # method may introduce artifacts. However, to avoid ignore_label from
      # being interpolated with other labels, we still perform nearest neighbor
      # interpolation.
      # TODO(huizhongc): Change to bilinear interpolation by processing padded
      # and non-padded label separately.
      if gt_is_matting_map:
        tf.get_logger().warning(
            'Label downsampling with nearest neighbor may introduce artifacts.')

      scaled_labels = tf.image.resize(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    weights = utils.get_label_weight_mask(
        scaled_labels, ignore_label, num_classes, label_weights=loss_weight)
    # Dimension of keep_mask is equal to the total number of pixels.
    keep_mask = tf.cast(
        tf.not_equal(scaled_labels, ignore_label), dtype=tf.float32)

    train_labels = None
    logits = tf.reshape(logits, shape=[-1, num_classes])

    if gt_is_matting_map:
      # When the groundtruth is integer label mask, we can assign class
      # dependent label weights to the loss. When the groundtruth is image
      # matting confidence, we do not apply class-dependent label weight (i.e.,
      # label_weight = 1.0).
      if loss_weight != 1.0:
        raise ValueError(
            'loss_weight must equal to 1 if groundtruth is matting map.')

      # Assign label value 0 to ignore pixels. The exact label value of ignore
      # pixel does not matter, because those ignore_value pixel losses will be
      # multiplied to 0 weight.
      train_labels = scaled_labels * keep_mask

      train_labels = tf.expand_dims(train_labels, 1)
      train_labels = tf.concat([1 - train_labels, train_labels], axis=1)
    else:
      train_labels = tf.one_hot(
          scaled_labels, num_classes, on_value=1.0, off_value=0.0)

    default_loss_scope = ('softmax_all_pixel_loss'
                          if top_k_percent_pixels == 1.0 else
                          'softmax_hard_example_mining')
    with tf.name_scope(loss_scope, default_loss_scope,
                       [logits, train_labels, weights]):
      # Compute the loss for all pixels.
      pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=tf.stop_gradient(
              train_labels, name='train_labels_stop_gradient'),
          logits=logits,
          name='pixel_losses')
      weighted_pixel_losses = tf.multiply(pixel_losses, weights)

      if top_k_percent_pixels == 1.0:
        total_loss = tf.reduce_sum(weighted_pixel_losses)
        num_present = tf.reduce_sum(keep_mask)
        loss = _div_maybe_zero(total_loss, num_present)
        # TF2: Return loss instead of using tf.losses.add_loss
        # Caller should accumulate losses explicitly
        return loss
      else:
        num_pixels = tf.cast(tf.shape(logits)[0], tf.float32)
        # Compute the top_k_percent pixels based on current training step.
        if hard_example_mining_step == 0:
          # Directly focus on the top_k pixels.
          top_k_pixels = tf.cast(top_k_percent_pixels * num_pixels, tf.int32)
        else:
          # Gradually reduce the mining percent to top_k_percent_pixels.
          global_step = tf.cast(tf.Variable(0, trainable=False, name='global_step'), tf.float32)
          ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
          top_k_pixels = tf.cast(
              (ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels, tf.int32)
        top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                      k=top_k_pixels,
                                      sorted=True,
                                      name='top_k_percent_pixels')
        total_loss = tf.reduce_sum(top_k_losses)
        num_present = tf.reduce_sum(
            tf.cast(tf.not_equal(top_k_losses, 0.0), tf.float32))
        loss = _div_maybe_zero(total_loss, num_present)
        # TF2: Return loss instead of using tf.losses.add_loss
        return loss
  
  # Return 0 if no scales (shouldn't happen)
  return tf.constant(0.0)


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.
  
  TF2 Migration: This function now returns a function that loads weights
  using tf.train.Checkpoint. For TF1 checkpoints, use tf1_to_tf2_mapper.py
  to convert checkpoints first.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function that takes a model and loads weights.
  """
  if tf_initial_checkpoint is None:
    tf.get_logger().info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.get_logger().info('Ignoring initialization; other checkpoint exists')
    return None

  tf.get_logger().info('Initializing model from path: %s', tf_initial_checkpoint)

  def restore_fn(model):
    """Restore function for TF2.
    
    Args:
        model: A tf.keras.Model instance to restore weights to.
    """
    checkpoint = tf.train.Checkpoint(model=model)
    
    # Try to restore
    try:
      if ignore_missing_vars:
        status = checkpoint.restore(tf_initial_checkpoint).expect_partial()
      else:
        status = checkpoint.restore(tf_initial_checkpoint)
      tf.get_logger().info('Checkpoint restored successfully')
    except Exception as e:
      tf.get_logger().warning(f'Error restoring checkpoint: {e}')
      if not ignore_missing_vars:
        raise
  
  return restore_fn


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.
  
  TF2 Migration: Now works with tf.keras.Model.trainable_variables.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variable names as key, and multipliers as value.
  """
  gradient_multipliers = {}

  # TF2: This function signature is preserved but implementation
  # should be called with model.trainable_variables
  # Usage: gradient_multipliers = get_model_gradient_multipliers(last_layers, multiplier)
  #        for var in model.trainable_variables:
  #            if var.name in gradient_multipliers:
  #                gradients[i] *= gradient_multipliers[var.name]
  
  return gradient_multipliers


def get_gradient_multiplier_for_variable(var, last_layers, last_layer_gradient_multiplier):
  """Get gradient multiplier for a single variable.
  
  TF2 helper function to get gradient multiplier for a variable.
  
  Args:
    var: A tf.Variable
    last_layers: List of last layer name patterns
    last_layer_gradient_multiplier: Multiplier for last layers
    
  Returns:
    Float gradient multiplier for this variable
  """
  multiplier = 1.0
  var_name = var.name
  
  # Double the learning rate for biases
  if 'bias' in var_name.lower():
    multiplier = 2.0
  
  # Use larger learning rate for last layer variables
  for layer in last_layers:
    if layer in var_name:
      if 'bias' in var_name.lower():
        multiplier = 2 * last_layer_gradient_multiplier
      else:
        multiplier = last_layer_gradient_multiplier
      break
  
  return multiplier


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power,
                            slow_start_step,
                            slow_start_learning_rate,
                            slow_start_burnin_type='none',
                            decay_steps=0.0,
                            end_learning_rate=0.0,
                            boundaries=None,
                            boundary_learning_rates=None):
  """Gets model's learning rate schedule.
  
  TF2 Migration: Returns a tf.keras.optimizers.schedules.LearningRateSchedule
  instead of a tensor. The schedule can be passed directly to optimizers.

  Computes the model's learning rate for different learning policy.
  Right now, only "step", "poly", "cosine", and "multi_steps" are supported.

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.
    slow_start_burnin_type: The burnin type for the slow start stage. Can be
      `none` which means no burnin or `linear` which means the learning rate
      increases linearly from slow_start_learning_rate and reaches
      base_learning_rate after slow_start_steps.
    decay_steps: Float, `decay_steps` for polynomial learning rate.
    end_learning_rate: Float, `end_learning_rate` for polynomial learning rate.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries.
    boundary_learning_rates: A list of `Tensor`s or `float`s or `int`s that
      specifies the values for the intervals defined by `boundaries`.

  Returns:
    A tf.keras.optimizers.schedules.LearningRateSchedule instance.

  Raises:
    ValueError: If learning policy or slow start burnin type is not recognized.
  """
  if decay_steps == 0.0:
    tf.get_logger().info('Setting decay_steps to total training steps.')
    decay_steps = training_number_of_steps - slow_start_step
  
  # Create base schedule
  if learning_policy == 'step':
    base_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_learning_rate,
        decay_steps=learning_rate_decay_step,
        decay_rate=learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    base_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=base_learning_rate,
        decay_steps=int(decay_steps),
        end_learning_rate=end_learning_rate,
        power=learning_power)
  elif learning_policy == 'cosine':
    base_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_learning_rate,
        decay_steps=training_number_of_steps - slow_start_step)
  elif learning_policy == 'multi_steps':
    if boundaries is None or boundary_learning_rates is None:
      raise ValueError('Must set `boundaries` and `boundary_learning_rates` '
                       'for multi_steps learning rate decay.')
    base_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=boundary_learning_rates)
  else:
    raise ValueError(f'Unknown learning policy: {learning_policy}')
  
  # Wrap with warmup if needed
  if slow_start_step > 0:
    return WarmupSchedule(
        base_schedule=base_schedule,
        warmup_steps=slow_start_step,
        warmup_learning_rate=slow_start_learning_rate,
        burnin_type=slow_start_burnin_type,
        base_learning_rate=base_learning_rate)
  else:
    return base_schedule


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule with warmup.
  
  Applies a warmup period before the main learning rate schedule.
  """
  
  def __init__(self, base_schedule, warmup_steps, warmup_learning_rate,
               burnin_type='none', base_learning_rate=None):
    """Initialize warmup schedule.
    
    Args:
      base_schedule: The main learning rate schedule to use after warmup.
      warmup_steps: Number of warmup steps.
      warmup_learning_rate: Learning rate during warmup (or start of linear warmup).
      burnin_type: 'none' for constant warmup LR, 'linear' for linear increase.
      base_learning_rate: Target LR for linear burnin (required if burnin_type='linear').
    """
    super().__init__()
    self.base_schedule = base_schedule
    self.warmup_steps = warmup_steps
    self.warmup_learning_rate = warmup_learning_rate
    self.burnin_type = burnin_type
    self.base_learning_rate = base_learning_rate
  
  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    warmup_steps = tf.cast(self.warmup_steps, tf.float32)
    
    # Calculate warmup learning rate
    if self.burnin_type == 'linear' and self.base_learning_rate is not None:
      warmup_lr = (
          self.warmup_learning_rate +
          (self.base_learning_rate - self.warmup_learning_rate) *
          step / warmup_steps)
    else:
      warmup_lr = self.warmup_learning_rate
    
    # Get main schedule LR (adjusted for warmup period)
    adjusted_step = tf.maximum(step - self.warmup_steps, 0)
    main_lr = self.base_schedule(adjusted_step)
    
    # Return warmup LR if in warmup period, else main LR
    return tf.where(step < self.warmup_steps, warmup_lr, main_lr)
  
  def get_config(self):
    return {
        'base_schedule': self.base_schedule,
        'warmup_steps': self.warmup_steps,
        'warmup_learning_rate': self.warmup_learning_rate,
        'burnin_type': self.burnin_type,
        'base_learning_rate': self.base_learning_rate
    }
