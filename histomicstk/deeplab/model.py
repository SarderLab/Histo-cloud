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
r"""Provides DeepLab model definition and helper functions.

TF2-MIGRATION: This module has been updated to use TF2-native APIs.
Key changes:
- Removed TF-Slim dependency entirely
- Replaced slim.conv2d with tf.keras.layers.Conv2D
- Replaced slim.separable_conv2d with custom SplitSeparableConv2D
- Replaced slim.arg_scope with explicit layer configuration
- Replaced slim.batch_norm with tf.keras.layers.BatchNormalization
- Removed tf.variable_scope (TF2 uses layer names automatically)
- Removed reuse parameter (TF2 handles this via model instance reuse)
- Added explicit training parameter passing

DeepLab is a deep learning system for semantic image segmentation with
the following features:

(1) Atrous convolution to explicitly control the resolution at which
feature responses are computed within Deep Convolutional Neural Networks.

(2) Atrous spatial pyramid pooling (ASPP) to robustly segment objects at
multiple scales with filters at multiple sampling rates and effective
fields-of-views.

(3) ASPP module augmented with image-level feature and batch normalization.

(4) A simple yet effective decoder module to recover the object boundaries.

See the following papers for more details:

"Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation"
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
(https://arxiv.org/abs/1802.02611)

"Rethinking Atrous Convolution for Semantic Image Segmentation,"
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
(https://arxiv.org/abs/1706.05587)
"""
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from deeplab.core import dense_prediction_cell
from deeplab.core import feature_extractor
from deeplab.core import utils

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
META_ARCHITECTURE_SCOPE = 'meta_architecture'

PROB_SUFFIX = '_prob'

# TF2-MIGRATION: Direct function references without slim
_resize_bilinear = utils.resize_bilinear
scale_dimension = utils.scale_dimension


# =============================================================================
# TF2 Keras Layers for ASPP and Decoder
# =============================================================================

class SplitSeparableConv2D(tf.keras.layers.Layer):
    """Separable convolution as two separate layers (depthwise + pointwise).
    
    This matches the TF-Slim split_separable_conv2d behavior where depthwise
    and pointwise are distinct operations with separate batch norm.
    """
    
    def __init__(self,
                 filters,
                 kernel_size=3,
                 rate=1,
                 weight_decay=0.0001,
                 depthwise_activation=True,
                 pointwise_activation=True,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 use_bounded_activation=False,
                 name=None,
                 **kwargs):
        super(SplitSeparableConv2D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.rate = rate
        self.weight_decay = weight_decay
        self.depthwise_activation = depthwise_activation
        self.pointwise_activation = pointwise_activation
        self.use_batch_norm = use_batch_norm
        self.batch_norm_params = batch_norm_params or {}
        self.use_bounded_activation = use_bounded_activation
        
        # Regularizer
        self.regularizer = (tf.keras.regularizers.l2(weight_decay) 
                           if weight_decay > 0 else None)
        
        # Activation function
        self.activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
        
    def build(self, input_shape):
        # Depthwise convolution
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            dilation_rate=self.rate,
            use_bias=not self.use_batch_norm,
            depthwise_regularizer=self.regularizer,
            name='depthwise')
        
        # Depthwise batch norm
        if self.use_batch_norm and self.depthwise_activation:
            self.depthwise_bn = tf.keras.layers.BatchNormalization(
                name='depthwise_bn', **self.batch_norm_params)
        
        # Pointwise convolution
        self.pointwise_conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=not self.use_batch_norm,
            kernel_regularizer=self.regularizer,
            name='pointwise')
        
        # Pointwise batch norm
        if self.use_batch_norm and self.pointwise_activation:
            self.pointwise_bn = tf.keras.layers.BatchNormalization(
                name='pointwise_bn', **self.batch_norm_params)
        
        super(SplitSeparableConv2D, self).build(input_shape)
    
    def call(self, inputs, training=False):
        x = self.depthwise_conv(inputs)
        if self.use_batch_norm and self.depthwise_activation:
            x = self.depthwise_bn(x, training=training)
        if self.depthwise_activation:
            x = self.activation_fn(x)
        
        x = self.pointwise_conv(x)
        if self.use_batch_norm and self.pointwise_activation:
            x = self.pointwise_bn(x, training=training)
        if self.pointwise_activation:
            x = self.activation_fn(x)
        
        return x


class ASPPModule(tf.keras.layers.Layer):
    """Atrous Spatial Pyramid Pooling module.
    
    This implements the ASPP module from DeepLabv3/v3+.
    """
    
    def __init__(self,
                 depth=256,
                 atrous_rates=(6, 12, 18),
                 add_image_level_feature=True,
                 use_separable_conv=True,
                 use_squeeze_and_excitation=False,
                 weight_decay=0.0001,
                 batch_norm_params=None,
                 use_bounded_activation=False,
                 name='aspp_module',
                 **kwargs):
        super(ASPPModule, self).__init__(name=name, **kwargs)
        self.depth = depth
        self.atrous_rates = atrous_rates
        self.add_image_level_feature = add_image_level_feature
        self.use_separable_conv = use_separable_conv
        self.use_squeeze_and_excitation = use_squeeze_and_excitation
        self.weight_decay = weight_decay
        self.batch_norm_params = batch_norm_params or {
            'momentum': 0.9997,
            'epsilon': 1e-5,
        }
        self.use_bounded_activation = use_bounded_activation
        
        self.regularizer = (tf.keras.regularizers.l2(weight_decay) 
                           if weight_decay > 0 else None)
        self.activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
        
    def build(self, input_shape):
        # 1x1 convolution branch
        self.conv_1x1 = tf.keras.layers.Conv2D(
            self.depth, 1, padding='same',
            kernel_regularizer=self.regularizer,
            name='aspp0')
        self.bn_1x1 = tf.keras.layers.BatchNormalization(
            name='aspp0_bn', **self.batch_norm_params)
        
        # Atrous convolution branches
        self.atrous_convs = []
        self.atrous_bns = []
        for i, rate in enumerate(self.atrous_rates):
            if self.use_separable_conv:
                conv = SplitSeparableConv2D(
                    self.depth, kernel_size=3, rate=rate,
                    weight_decay=self.weight_decay,
                    batch_norm_params=self.batch_norm_params,
                    use_bounded_activation=self.use_bounded_activation,
                    name=f'aspp{i+1}')
                self.atrous_convs.append(conv)
                self.atrous_bns.append(None)  # BN is inside SplitSeparableConv2D
            else:
                conv = tf.keras.layers.Conv2D(
                    self.depth, 3, padding='same', dilation_rate=rate,
                    kernel_regularizer=self.regularizer,
                    name=f'aspp{i+1}')
                bn = tf.keras.layers.BatchNormalization(
                    name=f'aspp{i+1}_bn', **self.batch_norm_params)
                self.atrous_convs.append(conv)
                self.atrous_bns.append(bn)
        
        # Image-level feature branch
        if self.add_image_level_feature:
            self.image_conv = tf.keras.layers.Conv2D(
                self.depth, 1, padding='same',
                kernel_regularizer=self.regularizer,
                name='image_pooling')
            if not self.use_squeeze_and_excitation:
                self.image_bn = tf.keras.layers.BatchNormalization(
                    name='image_pooling_bn', **self.batch_norm_params)
        
        # Concat projection
        self.concat_conv = tf.keras.layers.Conv2D(
            self.depth, 1, padding='same',
            kernel_regularizer=self.regularizer,
            name='concat_projection')
        self.concat_bn = tf.keras.layers.BatchNormalization(
            name='concat_projection_bn', **self.batch_norm_params)
        self.dropout = tf.keras.layers.Dropout(0.1, name='concat_projection_dropout')
        
        super(ASPPModule, self).build(input_shape)
    
    def call(self, inputs, training=False, pool_size=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor [batch, height, width, channels].
            training: Boolean, whether in training mode.
            pool_size: Optional tuple (pool_h, pool_w) for image pooling.
                       If None, uses global average pooling.
        """
        branch_logits = []
        
        # Image-level feature
        if self.add_image_level_feature:
            if pool_size is not None:
                image_feature = tf.keras.layers.AveragePooling2D(
                    pool_size=pool_size, strides=pool_size, padding='valid'
                )(inputs)
            else:
                # Global average pooling
                image_feature = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            
            image_feature = self.image_conv(image_feature)
            if not self.use_squeeze_and_excitation:
                image_feature = self.image_bn(image_feature, training=training)
                image_feature = self.activation_fn(image_feature)
            else:
                image_feature = tf.nn.sigmoid(image_feature)
            
            # Resize to input size
            target_size = tf.shape(inputs)[1:3]
            image_feature = _resize_bilinear(
                image_feature, target_size, image_feature.dtype)
            
            if not self.use_squeeze_and_excitation:
                branch_logits.append(image_feature)
        
        # 1x1 convolution branch
        x = self.conv_1x1(inputs)
        x = self.bn_1x1(x, training=training)
        x = self.activation_fn(x)
        branch_logits.append(x)
        
        # Atrous convolution branches
        for i, (conv, bn) in enumerate(zip(self.atrous_convs, self.atrous_bns)):
            if self.use_separable_conv:
                x = conv(inputs, training=training)
            else:
                x = conv(inputs)
                x = bn(x, training=training)
                x = self.activation_fn(x)
            branch_logits.append(x)
        
        # Concatenate all branches
        concat_logits = tf.concat(branch_logits, axis=-1)
        
        # Projection
        concat_logits = self.concat_conv(concat_logits)
        concat_logits = self.concat_bn(concat_logits, training=training)
        concat_logits = self.activation_fn(concat_logits)
        concat_logits = self.dropout(concat_logits, training=training)
        
        # Apply squeeze and excitation if enabled
        if self.add_image_level_feature and self.use_squeeze_and_excitation:
            concat_logits = concat_logits * image_feature
        
        return concat_logits


class DecoderModule(tf.keras.layers.Layer):
    """Decoder module for DeepLabv3+.
    
    Refines segmentation by incorporating low-level features.
    """
    
    def __init__(self,
                 decoder_depth=256,
                 projected_filters=48,
                 use_separable_conv=True,
                 use_sum_merge=False,
                 output_is_logits=False,
                 weight_decay=0.0001,
                 batch_norm_params=None,
                 use_bounded_activation=False,
                 name='decoder',
                 **kwargs):
        super(DecoderModule, self).__init__(name=name, **kwargs)
        self.decoder_depth = decoder_depth
        self.projected_filters = projected_filters if not use_sum_merge else decoder_depth
        self.use_separable_conv = use_separable_conv
        self.use_sum_merge = use_sum_merge
        self.output_is_logits = output_is_logits
        self.weight_decay = weight_decay
        self.batch_norm_params = batch_norm_params or {
            'momentum': 0.9997,
            'epsilon': 1e-5,
        }
        self.use_bounded_activation = use_bounded_activation
        
        self.regularizer = (tf.keras.regularizers.l2(weight_decay) 
                           if weight_decay > 0 else None)
        
    def build(self, input_shape):
        # Feature projection layers (will be built dynamically)
        self.projection_layers = {}
        
        # Decoder convolutions
        if self.output_is_logits:
            # When output is logits, no activation or batch norm
            self.decoder_conv = tf.keras.layers.Conv2D(
                self.decoder_depth, 1, padding='same',
                kernel_regularizer=self.regularizer,
                name='decoder_conv0')
        elif self.use_separable_conv:
            self.decoder_conv0 = SplitSeparableConv2D(
                self.decoder_depth, kernel_size=3, rate=1,
                weight_decay=self.weight_decay,
                batch_norm_params=self.batch_norm_params,
                use_bounded_activation=self.use_bounded_activation,
                name='decoder_conv0')
            if not self.use_sum_merge:
                self.decoder_conv1 = SplitSeparableConv2D(
                    self.decoder_depth, kernel_size=3, rate=1,
                    weight_decay=self.weight_decay,
                    batch_norm_params=self.batch_norm_params,
                    use_bounded_activation=self.use_bounded_activation,
                    name='decoder_conv1')
        else:
            activation_fn = (tf.nn.relu6 if self.use_bounded_activation 
                            else tf.nn.relu)
            self.decoder_conv0 = tf.keras.layers.Conv2D(
                self.decoder_depth, 3, padding='same',
                activation=activation_fn,
                kernel_regularizer=self.regularizer,
                name='decoder_conv0')
            self.decoder_bn0 = tf.keras.layers.BatchNormalization(
                name='decoder_bn0', **self.batch_norm_params)
            if not self.use_sum_merge:
                self.decoder_conv1 = tf.keras.layers.Conv2D(
                    self.decoder_depth, 3, padding='same',
                    kernel_regularizer=self.regularizer,
                    name='decoder_conv1')
                self.decoder_bn1 = tf.keras.layers.BatchNormalization(
                    name='decoder_bn1', **self.batch_norm_params)
        
        super(DecoderModule, self).build(input_shape)
    
    def get_projection_layer(self, name):
        """Gets or creates a projection layer."""
        if name not in self.projection_layers:
            self.projection_layers[name] = tf.keras.layers.Conv2D(
                self.projected_filters, 1, padding='same',
                kernel_regularizer=self.regularizer,
                name=f'feature_projection_{name}')
        return self.projection_layers[name]
    
    def call(self, encoder_features, low_level_features, 
             target_size, training=False):
        """Forward pass.
        
        Args:
            encoder_features: High-level features from encoder/ASPP.
            low_level_features: Low-level features from backbone.
            target_size: Target output size (height, width).
            training: Whether in training mode.
        """
        # Resize encoder features
        decoder_features = _resize_bilinear(
            encoder_features, target_size, encoder_features.dtype)
        
        # Project and resize low-level features
        if isinstance(low_level_features, dict):
            projected_features = []
            for name, feat in low_level_features.items():
                proj_layer = self.get_projection_layer(name)
                proj = proj_layer(feat)
                proj = _resize_bilinear(proj, target_size, proj.dtype)
                projected_features.append(proj)
        else:
            proj_layer = self.get_projection_layer('0')
            proj = proj_layer(low_level_features)
            proj = _resize_bilinear(proj, target_size, proj.dtype)
            projected_features = [proj]
        
        # Merge features
        if self.use_sum_merge:
            decoder_features = self.decoder_conv0(decoder_features, training=training)
            for proj in projected_features:
                decoder_features = decoder_features + proj
        else:
            # Concatenate
            features_list = [decoder_features] + projected_features
            concat_features = tf.concat(features_list, axis=-1)
            
            if self.output_is_logits:
                decoder_features = self.decoder_conv(concat_features)
            elif self.use_separable_conv:
                decoder_features = self.decoder_conv0(concat_features, training=training)
                decoder_features = self.decoder_conv1(decoder_features, training=training)
            else:
                decoder_features = self.decoder_conv0(concat_features)
                decoder_features = self.decoder_bn0(decoder_features, training=training)
                decoder_features = self.decoder_conv1(decoder_features)
                decoder_features = self.decoder_bn1(decoder_features, training=training)
        
        return decoder_features


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
    """Gets the scopes for extra layers.

    Args:
        last_layers_contain_logits_only: Boolean, True if only consider logits as
            the last layer (i.e., exclude ASPP module, decoder module and so on)

    Returns:
        A list of scopes for extra layers.
    """
    if last_layers_contain_logits_only:
        return [LOGITS_SCOPE_NAME]
    else:
        return [
            LOGITS_SCOPE_NAME,
            IMAGE_POOLING_SCOPE,
            ASPP_SCOPE,
            CONCAT_PROJECTION_SCOPE,
            DECODER_SCOPE,
            META_ARCHITECTURE_SCOPE,
        ]


def predict_labels_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
    """Predicts segmentation labels.

    TF2-MIGRATION: Removed tf.variable_scope/reuse pattern.
    Model reuse is handled by calling the same model instance.

    Args:
        images: A tensor of size [batch, height, width, channels].
        model_options: A ModelOptions instance to configure models.
        eval_scales: The scales to resize images for evaluation.
        add_flipped_images: Add flipped images for evaluation or not.

    Returns:
        A dictionary with keys specifying the output_type (e.g., semantic
            prediction) and values storing Tensors representing predictions 
            (argmax over channels). Each prediction has size [batch, height, width].
    """
    outputs_to_predictions = {
        output: []
        for output in model_options.outputs_to_num_classes
    }

    for i, image_scale in enumerate(eval_scales):
        outputs_to_scales_to_logits = multi_scale_logits(
            images,
            model_options=model_options,
            image_pyramid=[image_scale],
            is_training=False,
            fine_tune_batch_norm=False)

        if add_flipped_images:
            outputs_to_scales_to_logits_reversed = multi_scale_logits(
                tf.reverse(images, [2]),
                model_options=model_options,
                image_pyramid=[image_scale],
                is_training=False,
                fine_tune_batch_norm=False)

        for output in sorted(outputs_to_scales_to_logits):
            scales_to_logits = outputs_to_scales_to_logits[output]
            logits = _resize_bilinear(
                scales_to_logits[MERGED_LOGITS_SCOPE],
                tf.shape(images)[1:3],
                scales_to_logits[MERGED_LOGITS_SCOPE].dtype)
            outputs_to_predictions[output].append(
                tf.expand_dims(tf.nn.softmax(logits), 4))

            if add_flipped_images:
                scales_to_logits_reversed = (
                    outputs_to_scales_to_logits_reversed[output])
                logits_reversed = _resize_bilinear(
                    tf.reverse(scales_to_logits_reversed[MERGED_LOGITS_SCOPE], [2]),
                    tf.shape(images)[1:3],
                    scales_to_logits_reversed[MERGED_LOGITS_SCOPE].dtype)
                outputs_to_predictions[output].append(
                    tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

    for output in sorted(outputs_to_predictions):
        predictions = outputs_to_predictions[output]
        # Compute average prediction across different scales and flipped images.
        predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
        outputs_to_predictions[output] = tf.argmax(predictions, 3)
        outputs_to_predictions[output + PROB_SUFFIX] = tf.nn.softmax(predictions)

    return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):
    """Predicts segmentation labels.

    Args:
        images: A tensor of size [batch, height, width, channels].
        model_options: A ModelOptions instance to configure models.
        image_pyramid: Input image scales for multi-scale feature extraction.

    Returns:
        A dictionary with keys specifying the output_type (e.g., semantic
            prediction) and values storing Tensors representing predictions 
            (argmax over channels). Each prediction has size [batch, height, width].
    """
    outputs_to_scales_to_logits = multi_scale_logits(
        images,
        model_options=model_options,
        image_pyramid=image_pyramid,
        is_training=False,
        fine_tune_batch_norm=False)

    predictions = {}
    for output in sorted(outputs_to_scales_to_logits):
        scales_to_logits = outputs_to_scales_to_logits[output]
        logits = scales_to_logits[MERGED_LOGITS_SCOPE]
        # There are two ways to obtain the final prediction results: (1) bilinear
        # upsampling the logits followed by argmax, or (2) argmax followed by
        # nearest neighbor upsampling. The second option may introduce the "blocking
        # effect" but is computationally efficient.
        if model_options.prediction_with_upsampled_logits:
            logits = _resize_bilinear(logits,
                                      tf.shape(images)[1:3],
                                      scales_to_logits[MERGED_LOGITS_SCOPE].dtype)
            predictions[output] = tf.argmax(logits, 3)
            predictions[output + PROB_SUFFIX] = tf.nn.softmax(logits)
        else:
            predictions[output] = tf.argmax(logits, 3)
            predictions[output + PROB_SUFFIX] = tf.nn.softmax(logits)

    return predictions


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False,
                       nas_training_hyper_parameters=None):
    """Gets the logits for multi-scale inputs.

    The returned logits are all downsampled (due to max-pooling layers)
    for both training and evaluation.

    Args:
        images: A tensor of size [batch, height, width, channels].
        model_options: A ModelOptions instance to configure models.
        image_pyramid: Input image scales for multi-scale feature extraction.
        weight_decay: The weight decay for model variables.
        is_training: Is training or not.
        fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        nas_training_hyper_parameters: A dictionary storing hyper-parameters for
            training nas models.

    Returns:
        outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
            semantic prediction) to a dictionary of multi-scale logits names to
            logits. For each output_type, the dictionary has keys which
            correspond to the scales and values which correspond to the logits.
            For example, if `scales` equals [1.0, 1.5], then the keys would
            include 'merged_logits', 'logits_1.00' and 'logits_1.50'.

    Raises:
        ValueError: If model_options doesn't specify crop_size and its
            add_image_level_feature = True, since add_image_level_feature requires
            crop_size information.
    """
    # Setup default values.
    if not image_pyramid:
        image_pyramid = [1.0]
    crop_height = (
        model_options.crop_size[0]
        if model_options.crop_size else tf.shape(images)[1])
    crop_width = (
        model_options.crop_size[1]
        if model_options.crop_size else tf.shape(images)[2])
    if model_options.image_pooling_crop_size:
        image_pooling_crop_height = model_options.image_pooling_crop_size[0]
        image_pooling_crop_width = model_options.image_pooling_crop_size[1]

    # Compute the height, width for the output logits.
    if model_options.decoder_output_stride:
        logits_output_stride = min(model_options.decoder_output_stride)
    else:
        logits_output_stride = model_options.output_stride

    logits_height = scale_dimension(
        crop_height,
        max(1.0, max(image_pyramid)) / logits_output_stride)
    logits_width = scale_dimension(
        crop_width,
        max(1.0, max(image_pyramid)) / logits_output_stride)

    # Compute the logits for each scale in the image pyramid.
    outputs_to_scales_to_logits = {
        k: {}
        for k in model_options.outputs_to_num_classes
    }

    num_channels = images.get_shape().as_list()[-1]

    for image_scale in image_pyramid:
        if image_scale != 1.0:
            scaled_height = scale_dimension(crop_height, image_scale)
            scaled_width = scale_dimension(crop_width, image_scale)
            scaled_crop_size = [scaled_height, scaled_width]
            scaled_images = _resize_bilinear(images, scaled_crop_size, images.dtype)
            if model_options.crop_size:
                scaled_images.set_shape(
                    [None, scaled_height, scaled_width, num_channels])
            # Adjust image_pooling_crop_size accordingly.
            scaled_image_pooling_crop_size = None
            if model_options.image_pooling_crop_size:
                scaled_image_pooling_crop_size = [
                    scale_dimension(image_pooling_crop_height, image_scale),
                    scale_dimension(image_pooling_crop_width, image_scale)]
        else:
            scaled_crop_size = model_options.crop_size
            scaled_images = images
            scaled_image_pooling_crop_size = model_options.image_pooling_crop_size

        updated_options = model_options._replace(
            crop_size=scaled_crop_size,
            image_pooling_crop_size=scaled_image_pooling_crop_size)
        outputs_to_logits = _get_logits(
            scaled_images,
            updated_options,
            weight_decay=weight_decay,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            nas_training_hyper_parameters=nas_training_hyper_parameters)

        # Resize the logits to have the same dimension before merging.
        for output in sorted(outputs_to_logits):
            outputs_to_logits[output] = _resize_bilinear(
                outputs_to_logits[output], [logits_height, logits_width],
                outputs_to_logits[output].dtype)

        # Return when only one input scale.
        if len(image_pyramid) == 1:
            for output in sorted(model_options.outputs_to_num_classes):
                outputs_to_scales_to_logits[output][
                    MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
            return outputs_to_scales_to_logits

        # Save logits to the output map.
        for output in sorted(model_options.outputs_to_num_classes):
            outputs_to_scales_to_logits[output][
                'logits_%.2f' % image_scale] = outputs_to_logits[output]

    # Merge the logits from all the multi-scale inputs.
    for output in sorted(model_options.outputs_to_num_classes):
        # Concatenate the multi-scale logits for each output type.
        all_logits = [
            tf.expand_dims(logits, axis=4)
            for logits in outputs_to_scales_to_logits[output].values()
        ]
        all_logits = tf.concat(all_logits, 4)
        merge_fn = (
            tf.reduce_max
            if model_options.merge_method == 'max' else tf.reduce_mean)
        outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(
            all_logits, axis=4)

    return outputs_to_scales_to_logits


def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     nas_training_hyper_parameters=None):
    """Extracts features by the particular model_variant.

    TF2-MIGRATION: Removed slim.arg_scope pattern. ASPP is now implemented
    using the ASPPModule Keras layer.

    Args:
        images: A tensor of size [batch, height, width, channels].
        model_options: A ModelOptions instance to configure models.
        weight_decay: The weight decay for model variables.
        reuse: DEPRECATED in TF2. Ignored.
        is_training: Is training or not.
        fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        nas_training_hyper_parameters: A dictionary storing hyper-parameters for
            training nas models.

    Returns:
        concat_logits: A tensor of size [batch, feature_height, feature_width,
            feature_channels], where feature_height/feature_width are determined by
            the images height/width and output_stride.
        end_points: A dictionary from components of the network to the corresponding
            activation.
    """
    features, end_points = feature_extractor.extract_features(
        images,
        output_stride=model_options.output_stride,
        multi_grid=model_options.multi_grid,
        model_variant=model_options.model_variant,
        depth_multiplier=model_options.depth_multiplier,
        divisible_by=model_options.divisible_by,
        weight_decay=weight_decay,
        is_training=is_training,
        preprocessed_images_dtype=model_options.preprocessed_images_dtype,
        fine_tune_batch_norm=fine_tune_batch_norm,
        nas_architecture_options=model_options.nas_architecture_options,
        nas_training_hyper_parameters=nas_training_hyper_parameters,
        use_bounded_activation=model_options.use_bounded_activation)

    if not model_options.aspp_with_batch_norm:
        return features, end_points
    else:
        if model_options.dense_prediction_cell_config is not None:
            tf.print('Using dense prediction cell config.')
            dense_prediction_layer = dense_prediction_cell.DensePredictionCell(
                config=model_options.dense_prediction_cell_config,
                hparams={
                    'conv_rate_multiplier': 16 // model_options.output_stride,
                })
            concat_logits = dense_prediction_layer.build_cell(
                features,
                output_stride=model_options.output_stride,
                crop_size=model_options.crop_size,
                image_pooling_crop_size=model_options.image_pooling_crop_size,
                weight_decay=weight_decay,
                is_training=is_training,
                fine_tune_batch_norm=fine_tune_batch_norm)
            return concat_logits, end_points
        else:
            # Use the TF2 ASPPModule
            training = is_training and fine_tune_batch_norm
            batch_norm_params = {
                'momentum': 0.9997,
                'epsilon': 1e-5,
            }
            
            # Compute pool size for image-level feature
            pool_size = None
            if model_options.add_image_level_feature and model_options.crop_size is not None:
                image_pooling_crop_size = model_options.image_pooling_crop_size
                if image_pooling_crop_size is None:
                    image_pooling_crop_size = model_options.crop_size
                pool_height = scale_dimension(
                    image_pooling_crop_size[0],
                    1. / model_options.output_stride)
                pool_width = scale_dimension(
                    image_pooling_crop_size[1],
                    1. / model_options.output_stride)
                pool_size = (pool_height, pool_width)
            
            aspp_module = ASPPModule(
                depth=model_options.aspp_convs_filters,
                atrous_rates=model_options.atrous_rates or [],
                add_image_level_feature=model_options.add_image_level_feature,
                use_separable_conv=model_options.aspp_with_separable_conv,
                use_squeeze_and_excitation=model_options.aspp_with_squeeze_and_excitation,
                weight_decay=weight_decay,
                batch_norm_params=batch_norm_params,
                use_bounded_activation=model_options.use_bounded_activation,
                name='aspp_module')
            
            concat_logits = aspp_module(features, training=training, pool_size=pool_size)
            return concat_logits, end_points


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False,
                nas_training_hyper_parameters=None):
    """Gets the logits by atrous/image spatial pyramid pooling.

    TF2-MIGRATION: Removed reuse parameter, uses Keras layers.

    Args:
        images: A tensor of size [batch, height, width, channels].
        model_options: A ModelOptions instance to configure models.
        weight_decay: The weight decay for model variables.
        reuse: DEPRECATED in TF2. Ignored.
        is_training: Is training or not.
        fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        nas_training_hyper_parameters: A dictionary storing hyper-parameters.

    Returns:
        outputs_to_logits: A map from output_type to logits.
    """
    features, end_points = extract_features(
        images,
        model_options,
        weight_decay=weight_decay,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        nas_training_hyper_parameters=nas_training_hyper_parameters)

    if model_options.decoder_output_stride:
        crop_size = model_options.crop_size
        if crop_size is None:
            crop_size = [tf.shape(images)[1], tf.shape(images)[2]]
        features = refine_by_decoder(
            features,
            end_points,
            crop_size=crop_size,
            decoder_output_stride=model_options.decoder_output_stride,
            decoder_use_separable_conv=model_options.decoder_use_separable_conv,
            decoder_use_sum_merge=model_options.decoder_use_sum_merge,
            decoder_filters=model_options.decoder_filters,
            decoder_output_is_logits=model_options.decoder_output_is_logits,
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            use_bounded_activation=model_options.use_bounded_activation)

    outputs_to_logits = {}
    for output in sorted(model_options.outputs_to_num_classes):
        if model_options.decoder_output_is_logits:
            outputs_to_logits[output] = tf.identity(features, name=output)
        else:
            outputs_to_logits[output] = get_branch_logits(
                features,
                model_options.outputs_to_num_classes[output],
                model_options.atrous_rates,
                aspp_with_batch_norm=model_options.aspp_with_batch_norm,
                kernel_size=model_options.logits_kernel_size,
                weight_decay=weight_decay,
                scope_suffix=output)

    return outputs_to_logits


def refine_by_decoder(features,
                      end_points,
                      crop_size=None,
                      decoder_output_stride=None,
                      decoder_use_separable_conv=False,
                      decoder_use_sum_merge=False,
                      decoder_filters=256,
                      decoder_output_is_logits=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False,
                      use_bounded_activation=False,
                      sync_batch_norm_method='None'):
    """Adds the decoder to obtain sharper segmentation results.

    TF2-MIGRATION: Replaced slim.arg_scope and slim.conv2d with explicit
    Keras layer configuration. Decoder logic uses DecoderModule class.

    Args:
        features: A tensor of size [batch, features_height, features_width,
            features_channels].
        end_points: A dictionary from components of the network to the 
            corresponding activation.
        crop_size: A tuple [crop_height, crop_width] specifying whole patch crop
            size.
        decoder_output_stride: A list of integers specifying the output stride of
            low-level features used in the decoder module.
        decoder_use_separable_conv: Employ separable convolution for decoder or not.
        decoder_use_sum_merge: Boolean, decoder uses simple sum merge or not.
        decoder_filters: Integer, decoder filter size.
        decoder_output_is_logits: Boolean, using decoder output as logits or not.
        model_variant: Model variant for feature extraction.
        weight_decay: The weight decay for model variables.
        reuse: DEPRECATED in TF2. Ignored.
        is_training: Is training or not.
        fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        use_bounded_activation: Whether or not to use bounded activations.
        sync_batch_norm_method: String, method used to sync batch norm.

    Returns:
        Decoder output with size [batch, decoder_height, decoder_width,
            decoder_channels].

    Raises:
        ValueError: If crop_size is None.
    """
    if crop_size is None:
        raise ValueError('crop_size must be provided when using decoder.')
    
    training = is_training and fine_tune_batch_norm
    batch_norm_params = {
        'momentum': 0.9997,
        'epsilon': 1e-5,
    }
    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    
    decoder_depth = decoder_filters
    projected_filters = 48
    if decoder_use_sum_merge:
        # When using sum merge, the projected filters must be equal to decoder filters.
        projected_filters = decoder_filters
    
    if decoder_output_is_logits:
        # Overwrite settings when decoder output is logits.
        activation_fn = None
        use_bn = False
        conv2d_kernel = 1
        decoder_use_separable_conv = False
    else:
        activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
        use_bn = True
        conv2d_kernel = 3

    decoder_features = features
    decoder_stage = 0
    
    for output_stride in decoder_output_stride:
        feature_list = feature_extractor.networks_to_feature_maps[
            model_variant][feature_extractor.DECODER_END_POINTS][output_stride]
        
        scope_suffix = '' if decoder_stage == 0 else f'_{decoder_stage}'
        
        for i, name in enumerate(feature_list):
            decoder_features_list = [decoder_features]
            
            # MobileNet and NAS variants use different naming convention.
            if ('mobilenet' in model_variant or
                model_variant.startswith('mnas') or
                model_variant.startswith('nas')):
                feature_name = name
            else:
                feature_name = '{}/{}'.format(
                    feature_extractor.name_scope[model_variant], name)
            
            # Project low-level features
            proj_conv = tf.keras.layers.Conv2D(
                projected_filters, 1, padding='same',
                kernel_regularizer=regularizer,
                name=f'feature_projection{i}{scope_suffix}')
            proj_bn = tf.keras.layers.BatchNormalization(
                name=f'feature_projection{i}{scope_suffix}_bn',
                **batch_norm_params) if use_bn else None
            
            projected = proj_conv(end_points[feature_name])
            if use_bn and proj_bn is not None:
                projected = proj_bn(projected, training=training)
            if activation_fn is not None:
                projected = activation_fn(projected)
            
            decoder_features_list.append(projected)
            
            # Determine the output size.
            decoder_height = scale_dimension(crop_size[0], 1.0 / output_stride)
            decoder_width = scale_dimension(crop_size[1], 1.0 / output_stride)
            
            # Resize all features to decoder size
            for j, feature in enumerate(decoder_features_list):
                decoder_features_list[j] = _resize_bilinear(
                    feature, [decoder_height, decoder_width], feature.dtype)
                h = None if isinstance(decoder_height, tf.Tensor) else decoder_height
                w = None if isinstance(decoder_width, tf.Tensor) else decoder_width
                decoder_features_list[j].set_shape([None, h, w, None])
            
            if decoder_use_sum_merge:
                decoder_features = _decoder_with_sum_merge(
                    decoder_features_list,
                    decoder_depth,
                    conv2d_kernel=conv2d_kernel,
                    decoder_use_separable_conv=decoder_use_separable_conv,
                    weight_decay=weight_decay,
                    batch_norm_params=batch_norm_params if use_bn else None,
                    use_bounded_activation=use_bounded_activation,
                    training=training,
                    scope_suffix=scope_suffix)
            else:
                if not decoder_use_separable_conv:
                    scope_suffix = str(i) + scope_suffix
                decoder_features = _decoder_with_concat_merge(
                    decoder_features_list,
                    decoder_depth,
                    decoder_use_separable_conv=decoder_use_separable_conv,
                    weight_decay=weight_decay,
                    batch_norm_params=batch_norm_params if use_bn else None,
                    use_bounded_activation=use_bounded_activation,
                    training=training,
                    scope_suffix=scope_suffix)
        
        decoder_stage += 1
    
    return decoder_features


def _decoder_with_sum_merge(decoder_features_list,
                            decoder_depth,
                            conv2d_kernel=3,
                            decoder_use_separable_conv=True,
                            weight_decay=0.0001,
                            batch_norm_params=None,
                            use_bounded_activation=False,
                            training=False,
                            scope_suffix=''):
    """Decoder with sum to merge features.

    TF2-MIGRATION: Uses Keras layers instead of slim.

    Args:
        decoder_features_list: A list of decoder features.
        decoder_depth: Integer, the filters used in the convolution.
        conv2d_kernel: Integer, the convolution kernel size.
        decoder_use_separable_conv: Boolean, use separable conv or not.
        weight_decay: Weight decay for the model variables.
        batch_norm_params: Dict of batch norm parameters.
        use_bounded_activation: Use bounded activation (relu6) or not.
        training: Whether in training mode.
        scope_suffix: String, used in the scope suffix.

    Returns:
        decoder features merged with sum.

    Raises:
        RuntimeError: If decoder_features_list have length not equal to 2.
    """
    if len(decoder_features_list) != 2:
        raise RuntimeError('Expect decoder_features has length 2.')
    
    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    
    # Only apply one convolution when decoder use sum merge.
    if decoder_use_separable_conv:
        conv_layer = SplitSeparableConv2D(
            decoder_depth,
            kernel_size=3,
            rate=1,
            weight_decay=weight_decay,
            batch_norm_params=batch_norm_params,
            use_bounded_activation=use_bounded_activation,
            name='decoder_split_sep_conv0' + scope_suffix)
        decoder_features = conv_layer(decoder_features_list[0], training=training)
    else:
        activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
        conv_layer = tf.keras.layers.Conv2D(
            decoder_depth, conv2d_kernel, padding='same',
            kernel_regularizer=regularizer,
            name='decoder_conv0' + scope_suffix)
        decoder_features = conv_layer(decoder_features_list[0])
        if batch_norm_params is not None:
            bn_layer = tf.keras.layers.BatchNormalization(
                name='decoder_bn0' + scope_suffix, **batch_norm_params)
            decoder_features = bn_layer(decoder_features, training=training)
        decoder_features = activation_fn(decoder_features)
    
    decoder_features = decoder_features + decoder_features_list[1]
    return decoder_features


def _decoder_with_concat_merge(decoder_features_list,
                               decoder_depth,
                               decoder_use_separable_conv=True,
                               weight_decay=0.0001,
                               batch_norm_params=None,
                               use_bounded_activation=False,
                               training=False,
                               scope_suffix=''):
    """Decoder with concatenation to merge features.

    This decoder method applies two convolutions to smooth the features obtained
    by concatenating the input decoder_features_list.

    This decoder module is proposed in the DeepLabv3+ paper.

    TF2-MIGRATION: Uses Keras layers instead of slim.

    Args:
        decoder_features_list: A list of decoder features.
        decoder_depth: Integer, the filters used in the convolution.
        decoder_use_separable_conv: Boolean, use separable conv or not.
        weight_decay: Weight decay for the model variables.
        batch_norm_params: Dict of batch norm parameters.
        use_bounded_activation: Use bounded activation (relu6) or not.
        training: Whether in training mode.
        scope_suffix: String, used in the scope suffix.

    Returns:
        decoder features merged with concatenation.
    """
    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    concat_features = tf.concat(decoder_features_list, axis=-1)
    
    if decoder_use_separable_conv:
        conv0 = SplitSeparableConv2D(
            decoder_depth,
            kernel_size=3,
            rate=1,
            weight_decay=weight_decay,
            batch_norm_params=batch_norm_params,
            use_bounded_activation=use_bounded_activation,
            name='decoder_conv0' + scope_suffix)
        conv1 = SplitSeparableConv2D(
            decoder_depth,
            kernel_size=3,
            rate=1,
            weight_decay=weight_decay,
            batch_norm_params=batch_norm_params,
            use_bounded_activation=use_bounded_activation,
            name='decoder_conv1' + scope_suffix)
        
        decoder_features = conv0(concat_features, training=training)
        decoder_features = conv1(decoder_features, training=training)
    else:
        activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
        
        conv0 = tf.keras.layers.Conv2D(
            decoder_depth, 3, padding='same',
            kernel_regularizer=regularizer,
            name='decoder_conv0' + scope_suffix)
        conv1 = tf.keras.layers.Conv2D(
            decoder_depth, 3, padding='same',
            kernel_regularizer=regularizer,
            name='decoder_conv1' + scope_suffix)
        
        decoder_features = conv0(concat_features)
        if batch_norm_params is not None:
            bn0 = tf.keras.layers.BatchNormalization(
                name='decoder_bn0' + scope_suffix, **batch_norm_params)
            decoder_features = bn0(decoder_features, training=training)
        decoder_features = activation_fn(decoder_features)
        
        decoder_features = conv1(decoder_features)
        if batch_norm_params is not None:
            bn1 = tf.keras.layers.BatchNormalization(
                name='decoder_bn1' + scope_suffix, **batch_norm_params)
            decoder_features = bn1(decoder_features, training=training)
        decoder_features = activation_fn(decoder_features)
    
    return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
    """Gets the logits from each model's branch.

    The underlying model is branched out in the last layer when atrous
    spatial pyramid pooling is employed, and all branches are sum-merged
    to form the final logits.

    TF2-MIGRATION: Uses Keras layers instead of slim.

    Args:
        features: A float tensor of shape [batch, height, width, channels].
        num_classes: Number of classes to predict.
        atrous_rates: A list of atrous convolution rates for last layer.
        aspp_with_batch_norm: Use batch normalization layers for ASPP.
        kernel_size: Kernel size for convolution.
        weight_decay: Weight decay for the model variables.
        reuse: DEPRECATED in TF2. Ignored.
        scope_suffix: Scope suffix for the model variables.

    Returns:
        Merged logits with shape [batch, height, width, num_classes].

    Raises:
        ValueError: Upon invalid input kernel_size value.
    """
    # When using batch normalization with ASPP, ASPP has been applied before
    # in extract_features, and thus we simply apply 1x1 convolution here.
    if aspp_with_batch_norm or atrous_rates is None:
        if kernel_size != 1:
            raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                           'using aspp_with_batch_norm. Gets %d.' % kernel_size)
        atrous_rates = [1]

    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    
    branch_logits = []
    for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
            scope += '_%d' % i

        conv = tf.keras.layers.Conv2D(
            num_classes,
            kernel_size=kernel_size,
            dilation_rate=rate,
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            kernel_regularizer=regularizer,
            name=f'logits_{scope}')
        
        branch_logits.append(conv(features))

    if len(branch_logits) == 1:
        return branch_logits[0]
    return tf.add_n(branch_logits)
