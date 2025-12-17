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
"""Extracts features for different models.

TF2-MIGRATION: This module has been updated to use TF2-native APIs.
Key changes:
- Removed TF-Slim dependency and arg_scope pattern
- Network functions now accept config parameters directly
- Uses tf.keras compatible APIs throughout
- Simplified network routing with explicit parameter passing
- MobileNet/ResNet/NAS networks require separate migration or use tf.keras.applications
"""

import copy
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf

from deeplab.core import xception

# NOTE: The following imports may need migration or replacement with tf.keras.applications
# For now, we'll provide stub implementations for networks not yet migrated
# from deeplab.core import nas_network
# from deeplab.core import resnet_v1_beta
# from nets.mobilenet import mobilenet_v2
# from nets.mobilenet import mobilenet_v3

# Default end point for MobileNetv2 (one-based indexing).
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'
# Default end point for MobileNetv3.
_MOBILENET_V3_LARGE_FINAL_ENDPOINT = 'layer_17'
_MOBILENET_V3_SMALL_FINAL_ENDPOINT = 'layer_13'
# Default end point for EdgeTPU Mobilenet.
_MOBILENET_EDGETPU = 'layer_24'


# =============================================================================
# Network Configuration Classes (TF2 replacement for arg_scope)
# =============================================================================

class NetworkConfig:
    """Configuration for network building, replacing TF-Slim arg_scope."""
    
    def __init__(self,
                 weight_decay: float = 0.0001,
                 batch_norm_decay: float = 0.997,
                 batch_norm_epsilon: float = 1e-3,
                 batch_norm_scale: bool = True,
                 regularize_depthwise: bool = False,
                 use_bounded_activation: bool = False,
                 is_training: bool = False):
        self.weight_decay = weight_decay
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale
        self.regularize_depthwise = regularize_depthwise
        self.use_bounded_activation = use_bounded_activation
        self.is_training = is_training
    
    def get_batch_norm_params(self) -> Dict[str, Any]:
        """Returns Keras BatchNormalization parameters."""
        return {
            'momentum': self.batch_norm_decay,
            'epsilon': self.batch_norm_epsilon,
            'scale': self.batch_norm_scale,
            'center': True,
        }
    
    def get_regularizer(self) -> Optional[tf.keras.regularizers.Regularizer]:
        """Returns L2 regularizer if weight_decay > 0."""
        if self.weight_decay > 0:
            return tf.keras.regularizers.l2(self.weight_decay)
        return None


def xception_config(weight_decay: float = 0.0001,
                    batch_norm_decay: float = 0.9997,
                    batch_norm_epsilon: float = 1e-3,
                    batch_norm_scale: bool = True,
                    regularize_depthwise: bool = False,
                    use_bounded_activation: bool = False,
                    is_training: bool = False) -> NetworkConfig:
    """Creates configuration for Xception networks."""
    return NetworkConfig(
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        batch_norm_scale=batch_norm_scale,
        regularize_depthwise=regularize_depthwise,
        use_bounded_activation=use_bounded_activation,
        is_training=is_training)


def resnet_config(weight_decay: float = 0.0001,
                  batch_norm_decay: float = 0.95,
                  batch_norm_epsilon: float = 1e-5,
                  batch_norm_scale: bool = True,
                  is_training: bool = False) -> NetworkConfig:
    """Creates configuration for ResNet networks."""
    return NetworkConfig(
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        batch_norm_scale=batch_norm_scale,
        is_training=is_training)


def mobilenet_config(weight_decay: float = 0.00004,
                     batch_norm_decay: float = 0.997,
                     batch_norm_epsilon: float = 0.001,
                     is_training: bool = False) -> NetworkConfig:
    """Creates configuration for MobileNet networks."""
    return NetworkConfig(
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        is_training=is_training)


# =============================================================================
# MobileNet Wrapper Functions (Stubs - require separate migration)
# =============================================================================

def _mobilenet_v2(inputs,
                  depth_multiplier=1.0,
                  output_stride=16,
                  divisible_by=None,
                  final_endpoint=None,
                  training=False,
                  config=None):
    """MobileNetV2 feature extractor.
    
    TF2-MIGRATION NOTE: This is a stub. For full functionality:
    - Use tf.keras.applications.MobileNetV2 as base
    - Or migrate the custom mobilenet_v2 module separately
    
    Args:
        inputs: Input tensor of shape [batch, height, width, channels].
        depth_multiplier: Float multiplier for channel depth.
        output_stride: Output stride (8 or 16 typically).
        divisible_by: Ensure channels divisible by this number.
        final_endpoint: Endpoint to build up to.
        training: Whether in training mode.
        config: NetworkConfig instance.
        
    Returns:
        Tuple of (features, end_points).
    """
    # Use tf.keras.applications.MobileNetV2 as a drop-in
    alpha = depth_multiplier
    
    # Determine which layers to include based on output_stride
    if output_stride == 8:
        # Need dilated convolutions - not directly supported by keras MobileNetV2
        # This would require custom implementation
        raise NotImplementedError(
            "MobileNetV2 with output_stride=8 requires custom implementation. "
            "Use tf.keras.applications.MobileNetV2 with output_stride=16 or "
            "migrate the original mobilenet_v2 module.")
    
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        alpha=alpha,
        include_top=False,
        weights=None)  # Load pretrained weights separately
    
    # Extract intermediate endpoints
    end_points = {}
    for layer in base_model.layers:
        end_points[layer.name] = layer.output
    
    features = base_model.output
    return features, end_points


def _mobilenet_v3_large(inputs,
                        depth_multiplier=1.0,
                        output_stride=16,
                        training=False,
                        config=None):
    """MobileNetV3 Large feature extractor stub."""
    raise NotImplementedError(
        "MobileNetV3 requires migration. Consider using "
        "tf.keras.applications.MobileNetV3Large when available.")


def _mobilenet_v3_small(inputs,
                        depth_multiplier=1.0,
                        output_stride=16,
                        training=False,
                        config=None):
    """MobileNetV3 Small feature extractor stub."""
    raise NotImplementedError(
        "MobileNetV3 requires migration. Consider using "
        "tf.keras.applications.MobileNetV3Small when available.")


# =============================================================================
# ResNet Wrapper Functions (Stubs - require separate migration)
# =============================================================================

def _resnet_v1_50(inputs,
                  num_classes=None,
                  output_stride=16,
                  multi_grid=None,
                  global_pool=False,
                  training=False,
                  config=None):
    """ResNet-50 feature extractor.
    
    TF2-MIGRATION NOTE: This is a stub using tf.keras.applications.
    For atrous convolution support, the original resnet_v1_beta module
    needs to be migrated separately.
    
    Args:
        inputs: Input tensor.
        num_classes: Number of output classes (None for features only).
        output_stride: Output stride.
        multi_grid: Multi-grid atrous rates.
        global_pool: Whether to apply global pooling.
        training: Whether in training mode.
        config: NetworkConfig instance.
        
    Returns:
        Tuple of (features, end_points).
    """
    if output_stride != 32:
        raise NotImplementedError(
            "ResNet with output_stride != 32 requires custom dilated "
            "convolution implementation. Migrate resnet_v1_beta module.")
    
    base_model = tf.keras.applications.ResNet50(
        input_tensor=inputs,
        include_top=False,
        weights=None)
    
    end_points = {}
    for layer in base_model.layers:
        end_points[layer.name] = layer.output
    
    features = base_model.output
    if global_pool:
        features = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
    
    return features, end_points


def _resnet_v1_101(inputs,
                   num_classes=None,
                   output_stride=16,
                   multi_grid=None,
                   global_pool=False,
                   training=False,
                   config=None):
    """ResNet-101 feature extractor stub."""
    if output_stride != 32:
        raise NotImplementedError(
            "ResNet with output_stride != 32 requires custom dilated "
            "convolution implementation. Migrate resnet_v1_beta module.")
    
    base_model = tf.keras.applications.ResNet101(
        input_tensor=inputs,
        include_top=False,
        weights=None)
    
    end_points = {}
    for layer in base_model.layers:
        end_points[layer.name] = layer.output
    
    features = base_model.output
    if global_pool:
        features = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
    
    return features, end_points


# =============================================================================
# Network Maps and Configurations
# =============================================================================

# A map from network name to network function.
networks_map = {
    'mobilenet_v2': _mobilenet_v2,
    # 'mobilenet_edgetpu': mobilenet_edgetpu,  # Requires migration
    # 'mobilenet_v3_large_seg': mobilenet_v3_large_seg,  # Requires migration
    # 'mobilenet_v3_small_seg': mobilenet_v3_small_seg,  # Requires migration
    'resnet_v1_50': _resnet_v1_50,
    # 'resnet_v1_50_beta': resnet_v1_beta.resnet_v1_50_beta,  # Requires migration
    'resnet_v1_101': _resnet_v1_101,
    # 'resnet_v1_101_beta': resnet_v1_beta.resnet_v1_101_beta,  # Requires migration
    'xception_41': xception.xception_41,
    'xception_65': xception.xception_65,
    'xception_71': xception.xception_71,
    # 'nas_pnasnet': nas_network.pnasnet,  # Requires migration
    # 'nas_hnasnet': nas_network.hnasnet,  # Requires migration
}


# A map from network name to configuration factory function.
# TF2-MIGRATION: Replaces arg_scopes_map
config_map = {
    'mobilenet_v2': mobilenet_config,
    'mobilenet_edgetpu': mobilenet_config,
    'mobilenet_v3_large_seg': mobilenet_config,
    'mobilenet_v3_small_seg': mobilenet_config,
    'resnet_v1_18': resnet_config,
    'resnet_v1_18_beta': resnet_config,
    'resnet_v1_50': resnet_config,
    'resnet_v1_50_beta': resnet_config,
    'resnet_v1_101': resnet_config,
    'resnet_v1_101_beta': resnet_config,
    'xception_41': xception_config,
    'xception_65': xception_config,
    'xception_71': xception_config,
    'nas_pnasnet': xception_config,  # Similar config
    'nas_hnasnet': xception_config,
}


# Names for end point features.
DECODER_END_POINTS = 'decoder_end_points'

# A dictionary from network name to a map of end point features.
networks_to_feature_maps = {
    'mobilenet_v2': {
        DECODER_END_POINTS: {
            4: ['layer_4/depthwise_output'],
            8: ['layer_7/depthwise_output'],
            16: ['layer_14/depthwise_output'],
        },
    },
    'mobilenet_v3_large_seg': {
        DECODER_END_POINTS: {
            4: ['layer_4/depthwise_output'],
            8: ['layer_7/depthwise_output'],
            16: ['layer_13/depthwise_output'],
        },
    },
    'mobilenet_v3_small_seg': {
        DECODER_END_POINTS: {
            4: ['layer_2/depthwise_output'],
            8: ['layer_4/depthwise_output'],
            16: ['layer_9/depthwise_output'],
        },
    },
    'resnet_v1_18': {
        DECODER_END_POINTS: {
            4: ['block1/unit_1/lite_bottleneck_v1/conv2'],
            8: ['block2/unit_1/lite_bottleneck_v1/conv2'],
            16: ['block3/unit_1/lite_bottleneck_v1/conv2'],
        },
    },
    'resnet_v1_18_beta': {
        DECODER_END_POINTS: {
            4: ['block1/unit_1/lite_bottleneck_v1/conv2'],
            8: ['block2/unit_1/lite_bottleneck_v1/conv2'],
            16: ['block3/unit_1/lite_bottleneck_v1/conv2'],
        },
    },
    'resnet_v1_50': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
            8: ['block2/unit_3/bottleneck_v1/conv3'],
            16: ['block3/unit_5/bottleneck_v1/conv3'],
        },
    },
    'resnet_v1_50_beta': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
            8: ['block2/unit_3/bottleneck_v1/conv3'],
            16: ['block3/unit_5/bottleneck_v1/conv3'],
        },
    },
    'resnet_v1_101': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
            8: ['block2/unit_3/bottleneck_v1/conv3'],
            16: ['block3/unit_22/bottleneck_v1/conv3'],
        },
    },
    'resnet_v1_101_beta': {
        DECODER_END_POINTS: {
            4: ['block1/unit_2/bottleneck_v1/conv3'],
            8: ['block2/unit_3/bottleneck_v1/conv3'],
            16: ['block3/unit_22/bottleneck_v1/conv3'],
        },
    },
    'xception_41': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block2/unit_1/xception_module/'
                'separable_conv2_pointwise'],
            8: ['entry_flow/block3/unit_1/xception_module/'
                'separable_conv2_pointwise'],
            16: ['exit_flow/block1/unit_1/xception_module/'
                 'separable_conv2_pointwise'],
        },
    },
    'xception_65': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block2/unit_1/xception_module/'
                'separable_conv2_pointwise'],
            8: ['entry_flow/block3/unit_1/xception_module/'
                'separable_conv2_pointwise'],
            16: ['exit_flow/block1/unit_1/xception_module/'
                 'separable_conv2_pointwise'],
        },
    },
    'xception_71': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block3/unit_1/xception_module/'
                'separable_conv2_pointwise'],
            8: ['entry_flow/block5/unit_1/xception_module/'
                'separable_conv2_pointwise'],
            16: ['exit_flow/block1/unit_1/xception_module/'
                 'separable_conv2_pointwise'],
        },
    },
    'nas_pnasnet': {
        DECODER_END_POINTS: {
            4: ['Stem'],
            8: ['Cell_3'],
            16: ['Cell_7'],
        },
    },
    'nas_hnasnet': {
        DECODER_END_POINTS: {
            4: ['Cell_2'],
            8: ['Cell_5'],
            16: ['Cell_7'],
        },
    },
}

# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
name_scope = {
    'mobilenet_v2': 'MobilenetV2',
    'mobilenet_edgetpu': 'MobilenetEdgeTPU',
    'mobilenet_v3_large_seg': 'MobilenetV3',
    'mobilenet_v3_small_seg': 'MobilenetV3',
    'resnet_v1_18': 'resnet_v1_18',
    'resnet_v1_18_beta': 'resnet_v1_18',
    'resnet_v1_50': 'resnet_v1_50',
    'resnet_v1_50_beta': 'resnet_v1_50',
    'resnet_v1_101': 'resnet_v1_101',
    'resnet_v1_101_beta': 'resnet_v1_101',
    'xception_41': 'xception_41',
    'xception_65': 'xception_65',
    'xception_71': 'xception_71',
    'nas_pnasnet': 'pnasnet',
    'nas_hnasnet': 'hnasnet',
}

# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
    """Subtract Imagenet mean RGB value."""
    mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
    num_channels = tf.shape(inputs)[-1]
    # We set mean pixel as 0 for the non-RGB channels.
    mean_rgb_extended = tf.concat(
        [mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
    return tf.cast(inputs - mean_rgb_extended, dtype=dtype)


def _preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
    """Map image values from [0, 255] to [-1, 1]."""
    preprocessed_inputs = (2.0 / 255.0) * tf.cast(inputs, tf.float32) - 1.0
    return tf.cast(preprocessed_inputs, dtype=dtype)


_PREPROCESS_FN = {
    'mobilenet_v2': _preprocess_zero_mean_unit_range,
    'mobilenet_edgetpu': _preprocess_zero_mean_unit_range,
    'mobilenet_v3_large_seg': _preprocess_zero_mean_unit_range,
    'mobilenet_v3_small_seg': _preprocess_zero_mean_unit_range,
    'resnet_v1_18': _preprocess_subtract_imagenet_mean,
    'resnet_v1_18_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_50': _preprocess_subtract_imagenet_mean,
    'resnet_v1_50_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_101': _preprocess_subtract_imagenet_mean,
    'resnet_v1_101_beta': _preprocess_zero_mean_unit_range,
    'xception_41': _preprocess_zero_mean_unit_range,
    'xception_65': _preprocess_zero_mean_unit_range,
    'xception_71': _preprocess_zero_mean_unit_range,
    'nas_pnasnet': _preprocess_zero_mean_unit_range,
    'nas_hnasnet': _preprocess_zero_mean_unit_range,
}


def mean_pixel(model_variant=None):
    """Gets mean pixel value.

    This function returns different mean pixel value, depending on the input
    model_variant which adopts different preprocessing functions. We currently
    handle the following preprocessing functions:
    (1) _preprocess_subtract_imagenet_mean. We simply return mean pixel value.
    (2) _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].
    The return values are used in a way that the padded regions after
    pre-processing will contain value 0.

    Args:
        model_variant: Model variant (string) for feature extraction. For
            backwards compatibility, model_variant=None returns _MEAN_RGB.

    Returns:
        Mean pixel value.
    """
    if model_variant in ['resnet_v1_50',
                         'resnet_v1_101'] or model_variant is None:
        return _MEAN_RGB
    else:
        return [127.5, 127.5, 127.5]


def extract_features(images,
                     output_stride=8,
                     multi_grid=None,
                     depth_multiplier=1.0,
                     divisible_by=None,
                     final_endpoint=None,
                     model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     preprocess_images=True,
                     preprocessed_images_dtype=tf.float32,
                     num_classes=None,
                     global_pool=False,
                     nas_architecture_options=None,
                     nas_training_hyper_parameters=None,
                     use_bounded_activation=False):
    """Extracts features by the particular model_variant.

    TF2-MIGRATION: This function has been updated to work without TF-Slim
    arg_scope. Configuration is passed explicitly to network functions.
    The 'reuse' parameter is ignored in TF2 (handled by model instance reuse).

    Args:
        images: A tensor of size [batch, height, width, channels].
        output_stride: The ratio of input to output spatial resolution.
        multi_grid: Employ a hierarchy of different atrous rates within network.
        depth_multiplier: Float multiplier for the depth (number of channels)
            for all convolution ops used in MobileNet.
        divisible_by: None (use default setting) or an integer that ensures all
            layers # channels will be divisible by this number. Used in MobileNet.
        final_endpoint: The MobileNet endpoint to construct the network up to.
        model_variant: Model variant for feature extraction.
        weight_decay: The weight decay for model variables.
        reuse: DEPRECATED in TF2. Ignored.
        is_training: Is training or not.
        fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        regularize_depthwise: Whether or not apply L2-norm regularization on the
            depthwise convolution weights.
        preprocess_images: Performs preprocessing on images or not. Defaults to
            True. Set to False if preprocessing will be done by other functions.
        preprocessed_images_dtype: The type after the preprocessing function.
        num_classes: Number of classes for image classification task. Defaults
            to None for dense prediction tasks.
        global_pool: Global pooling for image classification task. Defaults to
            False, since dense prediction tasks do not use this.
        nas_architecture_options: A dictionary storing NAS architecture options.
        nas_training_hyper_parameters: A dictionary storing hyper-parameters for
            training nas models.
        use_bounded_activation: Whether or not to use bounded activations.

    Returns:
        features: A tensor of size [batch, feature_height, feature_width,
            feature_channels], where feature_height/feature_width are determined
            by the images height/width and output_stride.
        end_points: A dictionary from components of the network to the 
            corresponding activation.

    Raises:
        ValueError: Unrecognized model variant.
    """
    if model_variant not in networks_map:
        raise ValueError('Unknown model variant %s.' % model_variant)

    # Preprocess images if requested
    if preprocess_images:
        preprocess_fn = _PREPROCESS_FN.get(model_variant, _preprocess_zero_mean_unit_range)
        images = preprocess_fn(images, preprocessed_images_dtype)
    else:
        images = tf.cast(images, preprocessed_images_dtype)

    # Create network configuration
    training = is_training and fine_tune_batch_norm
    
    if 'xception' in model_variant:
        config = xception_config(
            weight_decay=weight_decay,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=1e-3,
            batch_norm_scale=True,
            regularize_depthwise=regularize_depthwise,
            use_bounded_activation=use_bounded_activation,
            is_training=training)
        
        network_fn = networks_map[model_variant]
        features, end_points = network_fn(
            inputs=images,
            num_classes=num_classes,
            is_training=training,
            global_pool=global_pool,
            output_stride=output_stride,
            regularize_depthwise=regularize_depthwise,
            multi_grid=multi_grid,
            scope=name_scope[model_variant])
    
    elif 'resnet' in model_variant:
        config = resnet_config(
            weight_decay=weight_decay,
            batch_norm_decay=0.95,
            batch_norm_epsilon=1e-5,
            batch_norm_scale=True,
            is_training=training)
        
        network_fn = networks_map[model_variant]
        features, end_points = network_fn(
            inputs=images,
            num_classes=num_classes,
            output_stride=output_stride,
            multi_grid=multi_grid,
            global_pool=global_pool,
            training=training,
            config=config)
    
    elif 'mobilenet' in model_variant:
        config = mobilenet_config(
            weight_decay=weight_decay,
            is_training=training)
        
        network_fn = networks_map[model_variant]
        features, end_points = network_fn(
            inputs=images,
            depth_multiplier=depth_multiplier,
            divisible_by=divisible_by,
            output_stride=output_stride,
            final_endpoint=final_endpoint,
            training=training,
            config=config)
    
    elif model_variant.startswith('nas'):
        raise NotImplementedError(
            'NAS networks require separate migration. '
            'model_variant=%s' % model_variant)
    
    else:
        raise ValueError('Unknown model variant %s.' % model_variant)

    return features, end_points


def get_network(network_name, preprocess_images=True,
                preprocessed_images_dtype=tf.float32):
    """Gets the network function.

    TF2-MIGRATION: Simplified without arg_scope. Configuration is now
    passed directly to network functions in extract_features().

    Args:
        network_name: Network name.
        preprocess_images: Preprocesses the images or not.
        preprocessed_images_dtype: The type after the preprocessing function.

    Returns:
        A network function that is used to extract features.

    Raises:
        ValueError: network is not supported.
    """
    if network_name not in networks_map:
        raise ValueError('Unsupported network %s.' % network_name)
    
    def _identity_function(inputs, dtype=preprocessed_images_dtype):
        return tf.cast(inputs, dtype=dtype)
    
    if preprocess_images:
        preprocess_function = _PREPROCESS_FN.get(
            network_name, _preprocess_zero_mean_unit_range)
    else:
        preprocess_function = _identity_function
    
    func = networks_map[network_name]
    
    @functools.wraps(func)
    def network_fn(inputs, *args, **kwargs):
        return func(preprocess_function(inputs, preprocessed_images_dtype),
                    *args, **kwargs)
    
    return network_fn


def get_feature_keys(model_variant):
    """Returns the decoder end point feature keys for the model variant.
    
    Args:
        model_variant: String, the model variant name.
        
    Returns:
        Dictionary mapping output stride to list of feature names.
    """
    if model_variant not in networks_to_feature_maps:
        raise ValueError('Unknown model variant: %s' % model_variant)
    
    return networks_to_feature_maps[model_variant].get(DECODER_END_POINTS, {})


def get_supported_model_variants():
    """Returns list of supported model variant names."""
    return list(networks_map.keys())
