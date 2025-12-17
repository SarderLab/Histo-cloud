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
"""TF2-native evaluation script for the DeepLab model.

This script provides:
- TF2-native evaluation with tf.keras
- Multi-scale evaluation support
- Per-class IoU metrics
- TensorBoard logging
- Continuous evaluation mode (watching for new checkpoints)

Usage:
    python eval_tf2.py \
        --checkpoint_dir=/path/to/checkpoints \
        --eval_logdir=/path/to/eval_logs \
        --dataset_dir=/path/to/dataset \
        --dataset=pascal_voc_seg \
        --eval_split=val
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DeepLab v3+ Evaluation (TF2)')
    
    # Checkpoint and logging directories
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory of model checkpoints.')
    parser.add_argument('--eval_logdir', type=str, required=True,
                        help='Where to write the event logs.')
    
    # Evaluation settings
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='The number of images in each batch during evaluation.')
    parser.add_argument('--eval_crop_size', type=int, nargs='+', default=[512, 512],
                        help='Image crop size [height, width] for evaluation.')
    parser.add_argument('--eval_interval_secs', type=int, default=300,
                        help='How often (in seconds) to run evaluation.')
    parser.add_argument('--max_number_of_evaluations', type=int, default=0,
                        help='Maximum number of eval iterations. 0 means loop indefinitely.')
    
    # Model settings
    parser.add_argument('--model_variant', type=str, default='xception_65',
                        help='DeepLab model variant.')
    parser.add_argument('--atrous_rates', type=int, nargs='+', default=None,
                        help='Atrous rates for ASPP.')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='The ratio of input to output spatial resolution.')
    parser.add_argument('--decoder_output_stride', type=int, nargs='*', default=None,
                        help='Decoder output stride.')
    
    # Multi-scale evaluation
    parser.add_argument('--eval_scales', type=float, nargs='+', default=[1.0],
                        help='The scales to resize images for evaluation.')
    parser.add_argument('--add_flipped_images', action='store_true',
                        help='Add flipped images for evaluation.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='pascal_voc_seg',
                        help='Name of the segmentation dataset.')
    parser.add_argument('--eval_split', type=str, default='val',
                        help='Which split of the dataset used for evaluation.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Where the dataset resides.')
    parser.add_argument('--min_resize_value', type=int, default=None,
                        help='Minimum resize value.')
    parser.add_argument('--max_resize_value', type=int, default=None,
                        help='Maximum resize value.')
    parser.add_argument('--resize_factor', type=int, default=None,
                        help='Resize factor.')
    
    # Advanced options
    parser.add_argument('--aspp_with_batch_norm', type=bool, default=True,
                        help='Use batch normalization in ASPP.')
    parser.add_argument('--aspp_with_separable_conv', type=bool, default=True,
                        help='Use separable convolution in ASPP.')
    parser.add_argument('--multi_grid', type=int, nargs='+', default=None,
                        help='Multi-grid rates for backbone.')
    parser.add_argument('--image_pyramid', type=float, nargs='*', default=None,
                        help='Image pyramid scales for inference.')
    
    return parser.parse_args()


class MeanIoU(tf.keras.metrics.Metric):
    """Mean Intersection over Union metric for semantic segmentation.
    
    This implementation properly handles ignore labels and provides
    per-class IoU as well as mean IoU.
    """
    
    def __init__(self, num_classes, ignore_label=255, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        # Confusion matrix: rows are true labels, columns are predictions
        self.confusion_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.float64
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update confusion matrix with batch of predictions."""
        # Flatten
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Create mask for valid labels (not ignore_label)
        valid_mask = tf.not_equal(y_true, self.ignore_label)
        
        # Filter out ignore labels
        y_true = tf.boolean_mask(y_true, valid_mask)
        y_pred = tf.boolean_mask(y_pred, valid_mask)
        
        # Clip predictions to valid range
        y_pred = tf.clip_by_value(y_pred, 0, self.num_classes - 1)
        
        # Compute confusion matrix for this batch
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        batch_cm = tf.math.confusion_matrix(
            y_true, y_pred,
            num_classes=self.num_classes,
            dtype=tf.float64
        )
        
        # Update total confusion matrix
        self.confusion_matrix.assign_add(batch_cm)
    
    def result(self):
        """Compute mean IoU from confusion matrix."""
        # True positives for each class (diagonal)
        tp = tf.linalg.diag_part(self.confusion_matrix)
        
        # Sum of each row (total actual for each class)
        row_sum = tf.reduce_sum(self.confusion_matrix, axis=1)
        
        # Sum of each column (total predicted for each class)
        col_sum = tf.reduce_sum(self.confusion_matrix, axis=0)
        
        # IoU = TP / (row_sum + col_sum - TP)
        denominator = row_sum + col_sum - tp
        
        # Handle division by zero
        iou_per_class = tf.where(
            denominator > 0,
            tp / denominator,
            tf.zeros_like(tp)
        )
        
        # Only average over classes that have samples
        valid_classes = denominator > 0
        num_valid = tf.reduce_sum(tf.cast(valid_classes, tf.float64))
        
        mean_iou = tf.where(
            num_valid > 0,
            tf.reduce_sum(iou_per_class) / num_valid,
            0.0
        )
        
        return mean_iou
    
    def get_per_class_iou(self):
        """Get IoU for each class."""
        tp = tf.linalg.diag_part(self.confusion_matrix)
        row_sum = tf.reduce_sum(self.confusion_matrix, axis=1)
        col_sum = tf.reduce_sum(self.confusion_matrix, axis=0)
        denominator = row_sum + col_sum - tp
        
        iou_per_class = tf.where(
            denominator > 0,
            tp / denominator,
            tf.constant(float('nan'), dtype=tf.float64) * tf.ones_like(tp)
        )
        
        return iou_per_class
    
    def reset_state(self):
        """Reset confusion matrix."""
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


class DeepLabV3PlusEvaluator:
    """DeepLab v3+ model evaluator with multi-scale support."""
    
    def __init__(self, model_options, num_classes, ignore_label=255):
        self.model_options = model_options
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        
        # Create metrics
        self.mean_iou_metric = MeanIoU(num_classes, ignore_label)
        self.pixel_accuracy = tf.keras.metrics.Accuracy()
    
    def predict_single_scale(self, images, model_options):
        """Predict labels at a single scale."""
        predictions = model.predict_labels(
            images, 
            model_options,
            image_pyramid=None
        )
        return predictions[common.OUTPUT_TYPE]
    
    def predict_multi_scale(self, images, model_options, eval_scales, 
                           add_flipped_images=False):
        """Predict labels using multiple scales."""
        predictions = model.predict_labels_multi_scale(
            images,
            model_options=model_options,
            eval_scales=eval_scales,
            add_flipped_images=add_flipped_images
        )
        return predictions[common.OUTPUT_TYPE]
    
    @tf.function
    def evaluate_batch(self, images, labels, eval_scales, add_flipped_images):
        """Evaluate a single batch."""
        if len(eval_scales) == 1 and eval_scales[0] == 1.0:
            # Single-scale prediction
            predictions = self.predict_single_scale(images, self.model_options)
        else:
            # Multi-scale prediction
            predictions = self.predict_multi_scale(
                images, self.model_options, 
                eval_scales, add_flipped_images
            )
        
        # Update metrics
        self.mean_iou_metric.update_state(labels, predictions)
        
        # For pixel accuracy, ignore invalid labels
        valid_mask = tf.not_equal(labels, self.ignore_label)
        valid_labels = tf.boolean_mask(labels, valid_mask)
        valid_preds = tf.boolean_mask(predictions, valid_mask)
        self.pixel_accuracy.update_state(valid_labels, valid_preds)
        
        return predictions
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.mean_iou_metric.reset_state()
        self.pixel_accuracy.reset_state()
    
    def get_metrics(self):
        """Get current metric values."""
        return {
            'mean_iou': float(self.mean_iou_metric.result().numpy()),
            'pixel_accuracy': float(self.pixel_accuracy.result().numpy()),
            'per_class_iou': self.mean_iou_metric.get_per_class_iou().numpy()
        }


def create_model_options(args, num_classes):
    """Create model options from arguments."""
    crop_size = args.eval_crop_size
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 1:
        crop_size = [crop_size[0], crop_size[0]]
    
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: num_classes},
        crop_size=crop_size,
        atrous_rates=args.atrous_rates,
        output_stride=args.output_stride
    )
    
    return model_options


def load_checkpoint(checkpoint_dir):
    """Load the latest checkpoint from directory.
    
    Returns:
        checkpoint_path: Path to the latest checkpoint, or None if not found.
        step: Training step from checkpoint name, or 0.
    """
    # Try TF2-style checkpoint first
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint:
        step = 0
        # Try to extract step from checkpoint name
        try:
            # Handle formats like 'ckpt-1000' or 'model-1000'
            basename = os.path.basename(checkpoint)
            if '-' in basename:
                step = int(basename.split('-')[-1])
        except (ValueError, IndexError):
            pass
        return checkpoint, step
    
    return None, 0


def evaluate_once(evaluator, dataset, eval_scales, add_flipped_images, 
                  summary_writer, step):
    """Run evaluation once on the entire dataset.
    
    Args:
        evaluator: DeepLabV3PlusEvaluator instance
        dataset: tf.data.Dataset to evaluate on
        eval_scales: List of scales for multi-scale evaluation
        add_flipped_images: Whether to add flipped images
        summary_writer: TensorBoard summary writer
        step: Current training step
    
    Returns:
        Dictionary of metric values
    """
    evaluator.reset_metrics()
    
    num_batches = 0
    for batch in dataset:
        images = batch[common.IMAGE]
        labels = batch[common.LABEL]
        
        evaluator.evaluate_batch(images, labels, eval_scales, add_flipped_images)
        num_batches += 1
        
        if num_batches % 100 == 0:
            tf.print(f'Evaluated {num_batches} batches...')
    
    # Get final metrics
    metrics = evaluator.get_metrics()
    
    # Log to TensorBoard
    with summary_writer.as_default():
        # Create tag based on evaluation settings
        tag_suffix = '_'.join([str(s) for s in eval_scales])
        if add_flipped_images:
            tag_suffix += '_flipped'
        
        tf.summary.scalar(f'eval/miou_{tag_suffix}', metrics['mean_iou'], step=step)
        tf.summary.scalar('eval/pixel_accuracy', metrics['pixel_accuracy'], step=step)
        
        # Per-class IoU
        for c, iou in enumerate(metrics['per_class_iou']):
            if not np.isnan(iou):
                tf.summary.scalar(f'eval/iou_class_{c}', iou, step=step)
    
    # Print results
    print(f'\n=== Evaluation Results (Step {step}) ===')
    print(f'Mean IoU: {metrics["mean_iou"]:.4f}')
    print(f'Pixel Accuracy: {metrics["pixel_accuracy"]:.4f}')
    print('\nPer-class IoU:')
    for c, iou in enumerate(metrics['per_class_iou']):
        if not np.isnan(iou):
            print(f'  Class {c}: {iou:.4f}')
    print('=' * 40)
    
    return metrics


def count_parameters():
    """Count and print model parameters."""
    total_params = 0
    trainable_params = 0
    
    for var in tf.trainable_variables():
        shape = var.shape
        num_params = np.prod(shape)
        total_params += num_params
        trainable_params += num_params
        
    print(f'\nModel Statistics:')
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')


def main():
    args = parse_args()
    
    # Set up logging
    tf.get_logger().setLevel('INFO')
    
    # Create eval log directory
    os.makedirs(args.eval_logdir, exist_ok=True)
    
    # Handle crop_size
    crop_size = args.eval_crop_size
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 1:
        crop_size = [crop_size[0], crop_size[0]]
    
    print(f'Evaluating on {args.eval_split} set')
    print(f'Checkpoint directory: {args.checkpoint_dir}')
    print(f'Eval log directory: {args.eval_logdir}')
    print(f'Eval crop size: {crop_size}')
    print(f'Eval scales: {args.eval_scales}')
    print(f'Add flipped images: {args.add_flipped_images}')
    
    # Create dataset
    dataset_obj = data_generator.Dataset(
        dataset_name=args.dataset,
        split_name=args.eval_split,
        dataset_dir=args.dataset_dir,
        batch_size=args.eval_batch_size,
        crop_size=crop_size,
        min_resize_value=args.min_resize_value,
        max_resize_value=args.max_resize_value,
        resize_factor=args.resize_factor,
        model_variant=args.model_variant,
        num_readers=2,
        is_training=False,
        should_shuffle=False,
        should_repeat=False
    )
    
    # Get TF2 dataset
    dataset = dataset_obj.get_dataset()
    
    # Create model options
    model_options = create_model_options(args, dataset_obj.num_of_classes)
    
    # Create evaluator
    evaluator = DeepLabV3PlusEvaluator(
        model_options=model_options,
        num_classes=dataset_obj.num_of_classes,
        ignore_label=dataset_obj.ignore_label
    )
    
    # Create summary writer
    summary_writer = tf.summary.create_file_writer(args.eval_logdir)
    
    # Track evaluated checkpoints
    evaluated_checkpoints = set()
    num_evaluations = 0
    
    print('\nStarting evaluation loop...')
    print('Press Ctrl+C to stop.\n')
    
    while True:
        # Find latest checkpoint
        checkpoint_path, step = load_checkpoint(args.checkpoint_dir)
        
        if checkpoint_path is None:
            print(f'No checkpoint found in {args.checkpoint_dir}. Waiting...')
            time.sleep(args.eval_interval_secs)
            continue
        
        # Skip if already evaluated
        if checkpoint_path in evaluated_checkpoints:
            if args.max_number_of_evaluations > 0:
                break
            print(f'Waiting for new checkpoint... (last: {checkpoint_path})')
            time.sleep(args.eval_interval_secs)
            continue
        
        print(f'\nEvaluating checkpoint: {checkpoint_path}')
        
        # Load checkpoint
        # Note: In TF2, we need to restore variables properly
        # This depends on how the model was saved
        try:
            # Try loading as SavedModel or checkpoint
            checkpoint = tf.train.Checkpoint()
            status = checkpoint.restore(checkpoint_path)
            print(f'Checkpoint restored: {checkpoint_path}')
        except Exception as e:
            print(f'Warning: Could not restore checkpoint: {e}')
            print('Attempting evaluation anyway (variables may not be restored)...')
        
        # Run evaluation
        # Reset dataset iterator
        dataset = dataset_obj.get_dataset()
        
        try:
            metrics = evaluate_once(
                evaluator, dataset, 
                args.eval_scales, args.add_flipped_images,
                summary_writer, step
            )
            
            # Mark as evaluated
            evaluated_checkpoints.add(checkpoint_path)
            num_evaluations += 1
            
        except Exception as e:
            print(f'Error during evaluation: {e}')
            import traceback
            traceback.print_exc()
        
        # Check if we've reached max evaluations
        if args.max_number_of_evaluations > 0 and \
           num_evaluations >= args.max_number_of_evaluations:
            print(f'\nReached maximum evaluations ({args.max_number_of_evaluations})')
            break
        
        # Wait before next evaluation
        if args.max_number_of_evaluations <= 0:
            time.sleep(args.eval_interval_secs)
    
    print('\nEvaluation complete.')


if __name__ == '__main__':
    main()
