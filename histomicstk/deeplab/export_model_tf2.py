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
"""TF2-native model export script for DeepLab.

This script exports trained DeepLab models in TF2 formats:
- SavedModel: The standard TF2 format, suitable for TF Serving, TF Lite, etc.
- TFLite: Optimized format for mobile and edge deployment

Usage:
    python export_model_tf2.py \
        --checkpoint_path=/path/to/checkpoint \
        --export_path=/path/to/export \
        --export_format=saved_model \
        --num_classes=21 \
        --crop_size 513 513

Export formats:
    saved_model: TF2 SavedModel format (default)
    tflite: TensorFlow Lite format for mobile deployment
    both: Export both formats
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from deeplab import common
from deeplab import input_preprocess
from deeplab import model


# Input/output tensor names
INPUT_NAME = 'ImageTensor'
OUTPUT_NAME = 'SemanticPredictions'
OUTPUT_PROB_NAME = 'SemanticProbabilities'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepLab v3+ Model Export (TF2)')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to checkpoint directory or file.')
    parser.add_argument('--export_path', type=str, required=True,
                        help='Path to export the model.')
    parser.add_argument('--export_format', type=str, default='saved_model',
                        choices=['saved_model', 'tflite', 'both'],
                        help='Export format: saved_model, tflite, or both.')
    
    # Model settings
    parser.add_argument('--num_classes', type=int, default=21,
                        help='Number of segmentation classes.')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[513, 513],
                        help='Crop size [height, width].')
    parser.add_argument('--atrous_rates', type=int, nargs='+', default=None,
                        help='Atrous rates for ASPP.')
    parser.add_argument('--output_stride', type=int, default=8,
                        help='The ratio of input to output spatial resolution.')
    parser.add_argument('--model_variant', type=str, default='xception_65',
                        help='DeepLab model variant.')
    parser.add_argument('--decoder_output_stride', type=int, nargs='*', default=None,
                        help='Decoder output stride.')
    
    # Inference settings
    parser.add_argument('--inference_scales', type=float, nargs='+', default=[1.0],
                        help='Scales for multi-scale inference.')
    parser.add_argument('--add_flipped_images', action='store_true',
                        help='Add flipped images during inference.')
    parser.add_argument('--image_pyramid', type=float, nargs='*', default=None,
                        help='Image pyramid scales.')
    
    # Input preprocessing
    parser.add_argument('--min_resize_value', type=int, default=None,
                        help='Minimum resize value.')
    parser.add_argument('--max_resize_value', type=int, default=None,
                        help='Maximum resize value.')
    parser.add_argument('--resize_factor', type=int, default=None,
                        help='Resize factor.')
    
    # TFLite options
    parser.add_argument('--tflite_quantize', action='store_true',
                        help='Apply quantization for TFLite export.')
    parser.add_argument('--tflite_input_shape', type=int, nargs=4, default=None,
                        help='Fixed input shape for TFLite [batch, height, width, channels].')
    
    return parser.parse_args()


class DeepLabExportModel(tf.keras.Model):
    """Wrapper model for export with preprocessing included."""
    
    def __init__(self, model_options, crop_size, 
                 min_resize_value=None, max_resize_value=None, resize_factor=None,
                 inference_scales=(1.0,), add_flipped_images=False,
                 image_pyramid=None, **kwargs):
        super().__init__(**kwargs)
        self.model_options = model_options
        self.crop_size = crop_size
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.inference_scales = inference_scales
        self.add_flipped_images = add_flipped_images
        self.image_pyramid = image_pyramid
        self._single_scale = (tuple(inference_scales) == (1.0,))
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8)])
    def call(self, input_image):
        """Run inference on input image.
        
        Args:
            input_image: uint8 tensor with shape [1, height, width, 3]
            
        Returns:
            Dictionary with:
                - semantic_predictions: int32 tensor [1, height, width]
                - semantic_probabilities: float32 tensor [1, height, width, num_classes]
        """
        original_image_size = tf.shape(input_image)[1:3]
        
        # Preprocess image
        image = tf.squeeze(input_image, axis=0)  # [H, W, 3]
        image = tf.cast(image, tf.float32)
        
        # Apply preprocessing
        resized_image, processed_image, _ = input_preprocess.preprocess_image_and_label(
            image,
            label=None,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            resize_factor=self.resize_factor,
            is_training=False,
            model_variant=self.model_options.model_variant
        )
        resized_image_size = tf.shape(resized_image)[:2]
        
        # Expand back to batch
        processed_image = tf.expand_dims(processed_image, 0)
        
        # Run model prediction
        if self._single_scale:
            predictions = model.predict_labels(
                processed_image,
                model_options=self.model_options,
                image_pyramid=self.image_pyramid
            )
        else:
            predictions = model.predict_labels_multi_scale(
                processed_image,
                model_options=self.model_options,
                eval_scales=self.inference_scales,
                add_flipped_images=self.add_flipped_images
            )
        
        # Get semantic predictions and probabilities
        semantic_pred = predictions[common.OUTPUT_TYPE]
        semantic_prob = predictions.get(
            common.OUTPUT_TYPE + model.PROB_SUFFIX,
            tf.zeros([1, resized_image_size[0], resized_image_size[1], 
                     self.model_options.outputs_to_num_classes[common.OUTPUT_TYPE]])
        )
        
        # Crop to valid region
        semantic_pred = semantic_pred[:, :resized_image_size[0], :resized_image_size[1]]
        semantic_prob = semantic_prob[:, :resized_image_size[0], :resized_image_size[1]]
        
        # Resize back to original size
        semantic_pred = tf.expand_dims(semantic_pred, -1)  # [1, H, W, 1]
        semantic_pred = tf.image.resize(
            semantic_pred,
            original_image_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        semantic_pred = tf.cast(tf.squeeze(semantic_pred, -1), tf.int32)
        
        semantic_prob = tf.image.resize(
            semantic_prob,
            original_image_size,
            method=tf.image.ResizeMethod.BILINEAR
        )
        
        return {
            OUTPUT_NAME: semantic_pred,
            OUTPUT_PROB_NAME: semantic_prob
        }
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)])
    def predict_preprocessed(self, preprocessed_image):
        """Run inference on already preprocessed image (float32).
        
        This signature is useful when preprocessing is done externally.
        
        Args:
            preprocessed_image: float32 tensor [batch, height, width, 3]
            
        Returns:
            Dictionary with semantic predictions and probabilities.
        """
        if tuple(self.inference_scales) == (1.0,):
            predictions = model.predict_labels(
                preprocessed_image,
                model_options=self.model_options,
                image_pyramid=self.image_pyramid
            )
        else:
            predictions = model.predict_labels_multi_scale(
                preprocessed_image,
                model_options=self.model_options,
                eval_scales=self.inference_scales,
                add_flipped_images=self.add_flipped_images
            )
        
        semantic_pred = predictions[common.OUTPUT_TYPE]
        semantic_prob = predictions.get(
            common.OUTPUT_TYPE + model.PROB_SUFFIX,
            tf.zeros_like(semantic_pred, dtype=tf.float32)
        )
        
        return {
            OUTPUT_NAME: tf.cast(semantic_pred, tf.int32),
            OUTPUT_PROB_NAME: semantic_prob
        }


def create_model_options(args):
    """Create ModelOptions from arguments."""
    crop_size = args.crop_size
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 1:
        crop_size = [crop_size[0], crop_size[0]]
    
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: args.num_classes},
        crop_size=crop_size,
        atrous_rates=args.atrous_rates,
        output_stride=args.output_stride,
        model_variant=args.model_variant,
        decoder_output_stride=args.decoder_output_stride
    )
    
    return model_options


def load_checkpoint(export_model, checkpoint_path):
    """Load weights from checkpoint.
    
    Args:
        export_model: The DeepLabExportModel instance
        checkpoint_path: Path to checkpoint directory or file
        
    Returns:
        True if checkpoint was loaded successfully
    """
    # Try to find the latest checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if checkpoint_path is None:
            print(f'No checkpoint found in directory')
            return False
    
    print(f'Loading checkpoint: {checkpoint_path}')
    
    # Build the model by calling it with dummy input
    dummy_input = tf.zeros([1, 513, 513, 3], dtype=tf.uint8)
    _ = export_model(dummy_input)
    
    # Create checkpoint and restore
    checkpoint = tf.train.Checkpoint(model=export_model)
    
    try:
        status = checkpoint.restore(checkpoint_path)
        # Try to assert that weights were restored
        try:
            status.assert_existing_objects_matched()
            print('Checkpoint restored successfully (all objects matched)')
        except AssertionError:
            print('Warning: Some checkpoint objects may not have been matched')
        return True
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        
        # Try alternative: load with expect_partial
        try:
            status = checkpoint.restore(checkpoint_path).expect_partial()
            print('Checkpoint partially restored')
            return True
        except Exception as e2:
            print(f'Alternative loading also failed: {e2}')
            return False


def export_saved_model(export_model, export_path):
    """Export model as SavedModel.
    
    Args:
        export_model: The model to export
        export_path: Directory to save the model
    """
    print(f'Exporting SavedModel to: {export_path}')
    
    # Define signatures
    signatures = {
        'serving_default': export_model.call,
        'predict_preprocessed': export_model.predict_preprocessed
    }
    
    tf.saved_model.save(
        export_model,
        export_path,
        signatures=signatures
    )
    
    print(f'SavedModel exported successfully')
    print(f'  Input: {INPUT_NAME} (uint8, [1, None, None, 3])')
    print(f'  Output: {OUTPUT_NAME} (int32, [1, H, W])')
    print(f'  Output: {OUTPUT_PROB_NAME} (float32, [1, H, W, num_classes])')


def export_tflite(export_model, export_path, quantize=False, input_shape=None):
    """Export model as TFLite.
    
    Args:
        export_model: The model to export
        export_path: Path for the .tflite file
        quantize: Whether to apply quantization
        input_shape: Fixed input shape [batch, height, width, channels]
    """
    print(f'Exporting TFLite to: {export_path}')
    
    # For TFLite, we need a concrete function with fixed shape
    if input_shape is None:
        # Use default shape
        input_shape = [1, 513, 513, 3]
    
    # Create a concrete function with fixed input shape
    @tf.function(input_signature=[
        tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    ])
    def inference_fn(image):
        # Use predict_preprocessed for TFLite (expects float32 input)
        return export_model.predict_preprocessed(image)
    
    # Get concrete function
    concrete_func = inference_fn.get_concrete_function()
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    if quantize:
        print('Applying dynamic range quantization...')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Additional options for better compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS  # For ops not yet supported in TFLite
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(export_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f'TFLite model exported successfully')
        print(f'  Model size: {len(tflite_model) / (1024*1024):.2f} MB')
        print(f'  Input shape: {input_shape}')
    except Exception as e:
        print(f'Error converting to TFLite: {e}')
        print('Note: Some model operations may not be supported in TFLite.')
        raise


def main():
    args = parse_args()
    
    # Set up logging
    tf.get_logger().setLevel('INFO')
    
    print('=' * 60)
    print('DeepLab v3+ Model Export (TF2)')
    print('=' * 60)
    print(f'Checkpoint: {args.checkpoint_path}')
    print(f'Export path: {args.export_path}')
    print(f'Export format: {args.export_format}')
    print(f'Num classes: {args.num_classes}')
    print(f'Crop size: {args.crop_size}')
    print(f'Model variant: {args.model_variant}')
    print(f'Output stride: {args.output_stride}')
    print(f'Inference scales: {args.inference_scales}')
    print('=' * 60)
    
    # Create model options
    model_options = create_model_options(args)
    
    # Handle crop_size
    crop_size = args.crop_size
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 1:
        crop_size = [crop_size[0], crop_size[0]]
    
    # Create export model
    export_model = DeepLabExportModel(
        model_options=model_options,
        crop_size=crop_size,
        min_resize_value=args.min_resize_value,
        max_resize_value=args.max_resize_value,
        resize_factor=args.resize_factor,
        inference_scales=tuple(args.inference_scales),
        add_flipped_images=args.add_flipped_images,
        image_pyramid=args.image_pyramid
    )
    
    # Load checkpoint
    if not load_checkpoint(export_model, args.checkpoint_path):
        print('Warning: Checkpoint loading may have failed. Proceeding anyway...')
    
    # Create export directory
    export_dir = os.path.dirname(args.export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
    
    # Export based on format
    if args.export_format in ('saved_model', 'both'):
        saved_model_path = args.export_path
        if args.export_format == 'both':
            saved_model_path = os.path.join(args.export_path, 'saved_model')
        export_saved_model(export_model, saved_model_path)
    
    if args.export_format in ('tflite', 'both'):
        tflite_path = args.export_path
        if args.export_format == 'both':
            tflite_path = os.path.join(args.export_path, 'model.tflite')
        elif not tflite_path.endswith('.tflite'):
            tflite_path = tflite_path + '.tflite'
        
        export_tflite(
            export_model, tflite_path,
            quantize=args.tflite_quantize,
            input_shape=args.tflite_input_shape
        )
    
    print('\nExport complete!')


if __name__ == '__main__':
    main()
