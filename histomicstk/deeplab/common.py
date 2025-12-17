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
"""Provides configuration options that are common to scripts.

Common configuration from train/eval/vis/export_model.py are collected here.
TF2 Migration: Replaced tf.app.flags with a defaults dictionary approach.
"""
import collections
import copy
import json
import tensorflow as tf


# TF2 Migration: Replace tf.app.flags with a defaults dictionary.
# Users can pass explicit values to ModelOptions constructor instead of FLAGS.
class _DefaultConfig:
    """Container for default configuration values (replaces tf.app.flags)."""
    
    # Input preprocessing
    min_resize_value = None
    max_resize_value = None
    resize_factor = None
    keep_aspect_ratio = True
    
    # Model settings
    logits_kernel_size = 1
    model_variant = 'mobilenet_v2'
    image_pyramid = None
    add_image_level_feature = True
    image_pooling_crop_size = None
    image_pooling_stride = '1,1'
    aspp_with_batch_norm = True
    aspp_with_separable_conv = True
    multi_grid = None
    depth_multiplier = 1.0
    divisible_by = None
    decoder_output_stride = None
    decoder_use_separable_conv = True
    merge_method = 'max'
    prediction_with_upsampled_logits = False
    dense_prediction_cell_json = ''
    nas_stem_output_num_conv_filters = 20
    nas_use_classification_head = False
    nas_remove_os32_stride = False
    use_bounded_activation = False
    aspp_with_concat_projection = True
    aspp_with_squeeze_and_excitation = False
    aspp_convs_filters = 256
    decoder_use_sum_merge = False
    decoder_filters = 256
    decoder_output_is_logits = False
    image_se_uses_qsigmoid = False
    label_weights = None
    batch_norm_decay = 0.9997


# Global defaults instance - can be modified or replaced
DEFAULTS = _DefaultConfig()


# Backwards compatibility shim for code that might reference FLAGS
# Note: Deprecated, use DEFAULTS or pass explicit config to ModelOptions
class _FlagsShim:
    """Shim to provide backwards compatibility for FLAGS references."""
    def __getattr__(self, name):
        return getattr(DEFAULTS, name, None)

# Deprecated: Use DEFAULTS instead
FLAGS = _FlagsShim()


# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'preprocessed_images_dtype',
        'merge_method',
        'add_image_level_feature',
        'image_pooling_crop_size',
        'image_pooling_stride',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant',
        'depth_multiplier',
        'divisible_by',
        'prediction_with_upsampled_logits',
        'dense_prediction_cell_config',
        'nas_architecture_options',
        'use_bounded_activation',
        'aspp_with_concat_projection',
        'aspp_with_squeeze_and_excitation',
        'aspp_convs_filters',
        'decoder_use_sum_merge',
        'decoder_filters',
        'decoder_output_is_logits',
        'image_se_uses_qsigmoid',
        'label_weights',
        'sync_batch_norm_method',
        'batch_norm_decay',
    ])):
  """Immutable class to hold model options.
  
  TF2 Migration: This class now accepts all configuration via constructor kwargs
  instead of reading from tf.app.flags.FLAGS. Missing values fall back to DEFAULTS.
  """

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              atrous_rates=None,
              output_stride=8,
              preprocessed_images_dtype=tf.float32,
              # TF2: Allow explicit overrides instead of relying on FLAGS
              merge_method=None,
              add_image_level_feature=None,
              image_pooling_crop_size=None,
              image_pooling_stride=None,
              aspp_with_batch_norm=None,
              aspp_with_separable_conv=None,
              multi_grid=None,
              decoder_output_stride=None,
              decoder_use_separable_conv=None,
              logits_kernel_size=None,
              model_variant=None,
              depth_multiplier=None,
              divisible_by=None,
              prediction_with_upsampled_logits=None,
              dense_prediction_cell_json=None,
              nas_stem_output_num_conv_filters=None,
              nas_use_classification_head=None,
              nas_remove_os32_stride=None,
              use_bounded_activation=None,
              aspp_with_concat_projection=None,
              aspp_with_squeeze_and_excitation=None,
              aspp_convs_filters=None,
              decoder_use_sum_merge=None,
              decoder_filters=None,
              decoder_output_is_logits=None,
              image_se_uses_qsigmoid=None,
              label_weights=None,
              sync_batch_norm_method=None,
              batch_norm_decay=None):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: int size [size, size].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.
      preprocessed_images_dtype: The type after the preprocessing function.
      **All other args**: Override defaults. If None, falls back to DEFAULTS.

    Returns:
      A new ModelOptions instance.
    """
    # Helper to get value with fallback to DEFAULTS
    def _get(value, attr_name):
      if value is not None:
        return value
      return getattr(DEFAULTS, attr_name, None)
    
    # Parse dense_prediction_cell_config from JSON file if provided
    dense_prediction_cell_config = None
    json_path = dense_prediction_cell_json or getattr(DEFAULTS, 'dense_prediction_cell_json', '')
    if json_path:
      with tf.io.gfile.GFile(json_path, 'r') as f:
        dense_prediction_cell_config = json.load(f)
    
    # Parse decoder_output_stride
    _decoder_output_stride = decoder_output_stride
    if _decoder_output_stride is None:
      _decoder_output_stride = _get(None, 'decoder_output_stride')
    if _decoder_output_stride is not None:
      if isinstance(_decoder_output_stride, str):
        _decoder_output_stride = [int(x) for x in _decoder_output_stride.split(',')]
      elif not isinstance(_decoder_output_stride, list):
        _decoder_output_stride = list(_decoder_output_stride)
      else:
        _decoder_output_stride = [int(x) for x in _decoder_output_stride]
      if sorted(_decoder_output_stride, reverse=True) != _decoder_output_stride:
        raise ValueError('Decoder output stride need to be sorted in the '
                         'descending order.')
    
    # Parse image_pooling_crop_size
    _image_pooling_crop_size = image_pooling_crop_size
    if _image_pooling_crop_size is None:
      _image_pooling_crop_size = _get(None, 'image_pooling_crop_size')
    if _image_pooling_crop_size is not None:
      if isinstance(_image_pooling_crop_size, str):
        _image_pooling_crop_size = [int(x) for x in _image_pooling_crop_size.split(',')]
      else:
        _image_pooling_crop_size = [int(x) for x in _image_pooling_crop_size]
    
    # Parse image_pooling_stride
    _image_pooling_stride = image_pooling_stride
    if _image_pooling_stride is None:
      _image_pooling_stride = _get(None, 'image_pooling_stride')
    if _image_pooling_stride is None:
      _image_pooling_stride = [1, 1]
    elif isinstance(_image_pooling_stride, str):
      _image_pooling_stride = [int(x) for x in _image_pooling_stride.split(',')]
    else:
      _image_pooling_stride = [int(x) for x in _image_pooling_stride]
    
    # Handle label_weights
    _label_weights = label_weights
    if _label_weights is None:
      _label_weights = _get(None, 'label_weights')
    if _label_weights is None:
      _label_weights = 1.0
    
    # Build NAS architecture options
    nas_architecture_options = {
        'nas_stem_output_num_conv_filters': (
            nas_stem_output_num_conv_filters if nas_stem_output_num_conv_filters is not None
            else _get(20, 'nas_stem_output_num_conv_filters')),
        'nas_use_classification_head': (
            nas_use_classification_head if nas_use_classification_head is not None
            else _get(False, 'nas_use_classification_head')),
        'nas_remove_os32_stride': (
            nas_remove_os32_stride if nas_remove_os32_stride is not None
            else _get(False, 'nas_remove_os32_stride')),
    }
    
    return super(ModelOptions, cls).__new__(
        cls,
        outputs_to_num_classes,
        crop_size,
        atrous_rates,
        output_stride,
        preprocessed_images_dtype,
        _get(merge_method, 'merge_method') or 'max',
        _get(add_image_level_feature, 'add_image_level_feature') if add_image_level_feature is not None else True,
        _image_pooling_crop_size,
        _image_pooling_stride,
        _get(aspp_with_batch_norm, 'aspp_with_batch_norm') if aspp_with_batch_norm is not None else True,
        _get(aspp_with_separable_conv, 'aspp_with_separable_conv') if aspp_with_separable_conv is not None else True,
        _get(multi_grid, 'multi_grid'),
        _decoder_output_stride,
        _get(decoder_use_separable_conv, 'decoder_use_separable_conv') if decoder_use_separable_conv is not None else True,
        _get(logits_kernel_size, 'logits_kernel_size') or 1,
        _get(model_variant, 'model_variant') or 'mobilenet_v2',
        _get(depth_multiplier, 'depth_multiplier') or 1.0,
        _get(divisible_by, 'divisible_by'),
        _get(prediction_with_upsampled_logits, 'prediction_with_upsampled_logits') if prediction_with_upsampled_logits is not None else False,
        dense_prediction_cell_config,
        nas_architecture_options,
        _get(use_bounded_activation, 'use_bounded_activation') if use_bounded_activation is not None else False,
        _get(aspp_with_concat_projection, 'aspp_with_concat_projection') if aspp_with_concat_projection is not None else True,
        _get(aspp_with_squeeze_and_excitation, 'aspp_with_squeeze_and_excitation') if aspp_with_squeeze_and_excitation is not None else False,
        _get(aspp_convs_filters, 'aspp_convs_filters') or 256,
        _get(decoder_use_sum_merge, 'decoder_use_sum_merge') if decoder_use_sum_merge is not None else False,
        _get(decoder_filters, 'decoder_filters') or 256,
        _get(decoder_output_is_logits, 'decoder_output_is_logits') if decoder_output_is_logits is not None else False,
        _get(image_se_uses_qsigmoid, 'image_se_uses_qsigmoid') if image_se_uses_qsigmoid is not None else False,
        _label_weights,
        sync_batch_norm_method or 'None',
        _get(batch_norm_decay, 'batch_norm_decay') or 0.9997)

  def __deepcopy__(self, memo):
    return ModelOptions(copy.deepcopy(self.outputs_to_num_classes),
                        self.crop_size,
                        self.atrous_rates,
                        self.output_stride,
                        self.preprocessed_images_dtype)
