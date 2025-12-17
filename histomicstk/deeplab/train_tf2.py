#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""TF2-Native Training script for the DeepLab model.

TF2-MIGRATION: This is a complete rewrite of train.py using TF2-native APIs.
Key changes:
- Uses tf.keras.Model for the model
- Uses tf.GradientTape for custom training loop
- Uses tf.distribute.Strategy for multi-GPU training
- Uses tf.keras.optimizers instead of tf.train optimizers
- Uses argparse instead of tf.app.flags
- Uses tf.summary for TensorBoard logging
- Eager execution by default with @tf.function for performance

See model.py for more details and usage.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import tensorflow as tf

from deeplab import common
from deeplab import model
from deeplab.datasets import wsi_data_generator
from deeplab.core import feature_extractor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DeepLab model')
    
    # Multi-GPU settings
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use for training.')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training.')
    
    # Logging settings
    parser.add_argument('--train_logdir', type=str, required=True,
                        help='Where the checkpoint and logs are stored.')
    parser.add_argument('--log_steps', type=int, default=10,
                        help='Display logging information at every log_steps.')
    parser.add_argument('--save_interval_steps', type=int, default=1000,
                        help='How often, in steps, we save the model to disk.')
    parser.add_argument('--save_summaries_steps', type=int, default=100,
                        help='How often, in steps, we compute the summaries.')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['momentum', 'adam', 'adamw'],
                        help='Which optimizer to use.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='The base learning rate for model training.')
    parser.add_argument('--learning_policy', type=str, default='poly',
                        choices=['poly', 'step', 'cosine'],
                        help='Learning rate policy for training.')
    parser.add_argument('--learning_power', type=float, default=0.9,
                        help='The power value used in the poly learning policy.')
    parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                        help='End learning rate for polynomial/cosine schedule.')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The momentum value to use.')
    parser.add_argument('--weight_decay', type=float, default=0.00004,
                        help='The value of the weight decay for training.')
    
    # Training settings
    parser.add_argument('--training_number_of_steps', type=int, default=30000,
                        help='The number of steps used for training.')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='The number of images in each batch during training.')
    parser.add_argument('--train_crop_size', type=int, default=512,
                        help='Image crop size [size, size] during training.')
    parser.add_argument('--fine_tune_batch_norm', action='store_true',
                        help='Fine tune the batch norm parameters or not.')
    
    # Data augmentation
    parser.add_argument('--min_scale_factor', type=float, default=0.5,
                        help='Minimum scale factor for data augmentation.')
    parser.add_argument('--max_scale_factor', type=float, default=2.0,
                        help='Maximum scale factor for data augmentation.')
    parser.add_argument('--scale_factor_step_size', type=float, default=0.25,
                        help='Scale factor step size for data augmentation.')
    parser.add_argument('--augment_prob', type=float, default=0.0,
                        help='Probability that training data will be augmented.')
    
    # Model settings
    parser.add_argument('--model_variant', type=str, default='xception_65',
                        help='DeepLab model variant.')
    parser.add_argument('--atrous_rates', type=int, nargs='+', default=[6, 12, 18],
                        help='Atrous rates for atrous spatial pyramid pooling.')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='The ratio of input to output spatial resolution.')
    parser.add_argument('--decoder_output_stride', type=int, nargs='+', default=[4],
                        help='Decoder output stride.')
    
    # Checkpoint settings
    parser.add_argument('--tf_initial_checkpoint', type=str, default=None,
                        help='The initial checkpoint in tensorflow format.')
    parser.add_argument('--initialize_last_layer', action='store_true',
                        help='Initialize the last layer.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='wsi_dataset',
                        help='Name of the segmentation dataset.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Where the dataset reside.')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes in the dataset.')
    parser.add_argument('--ignore_label', type=int, default=255,
                        help='Int label of the data class to ignore during training.')
    parser.add_argument('--wsi_downsample', type=int, default=4,
                        help='Downsample rate of WSI used during training.')
    
    # Loss settings
    parser.add_argument('--top_k_percent_pixels', type=float, default=1.0,
                        help='The top k percent pixels used to compute loss.')
    parser.add_argument('--hard_example_mining_step', type=int, default=0,
                        help='Step when hard example mining kicks off.')
    
    return parser.parse_args()


class DeepLabV3PlusModel(tf.keras.Model):
    """DeepLab V3+ model as a Keras Model.
    
    This wraps the functional model components into a tf.keras.Model
    for easy training with custom training loops.
    """
    
    def __init__(self, 
                 num_classes: int,
                 model_variant: str = 'xception_65',
                 output_stride: int = 16,
                 atrous_rates: Tuple[int, ...] = (6, 12, 18),
                 decoder_output_stride: Tuple[int, ...] = (4,),
                 crop_size: Tuple[int, int] = (512, 512),
                 fine_tune_batch_norm: bool = True,
                 **kwargs):
        super(DeepLabV3PlusModel, self).__init__(**kwargs)
        
        self.num_classes = num_classes
        self.model_variant = model_variant
        self.output_stride = output_stride
        self.atrous_rates = atrous_rates
        self.decoder_output_stride = decoder_output_stride
        self.crop_size = crop_size
        self.fine_tune_batch_norm = fine_tune_batch_norm
        
        # Build model options
        self.model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: num_classes},
            crop_size=list(crop_size),
            atrous_rates=list(atrous_rates) if atrous_rates else None,
            output_stride=output_stride,
            decoder_output_stride=list(decoder_output_stride) if decoder_output_stride else None,
        )
    
    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: Input tensor [batch, height, width, channels].
            training: Whether in training mode.
            
        Returns:
            Dictionary of output logits.
        """
        outputs_to_scales_to_logits = model.multi_scale_logits(
            inputs,
            model_options=self.model_options,
            image_pyramid=[1.0],
            weight_decay=0.0,  # Handled by optimizer
            is_training=training,
            fine_tune_batch_norm=training and self.fine_tune_batch_norm)
        
        return {
            output: scales_to_logits[model.MERGED_LOGITS_SCOPE]
            for output, scales_to_logits in outputs_to_scales_to_logits.items()
        }


def create_learning_rate_schedule(args, total_steps):
    """Creates learning rate schedule.
    
    Args:
        args: Command line arguments.
        total_steps: Total number of training steps.
        
    Returns:
        Learning rate schedule.
    """
    if args.learning_policy == 'poly':
        schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=total_steps,
            end_learning_rate=args.end_learning_rate,
            power=args.learning_power)
    elif args.learning_policy == 'cosine':
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=total_steps,
            alpha=args.end_learning_rate / args.learning_rate)
    elif args.learning_policy == 'step':
        # Simple step decay
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=total_steps // 3,
            decay_rate=0.1,
            staircase=True)
    else:
        schedule = args.learning_rate
    
    # Add warmup if specified
    if args.warmup_steps > 0:
        schedule = WarmupSchedule(schedule, args.warmup_steps, args.learning_rate)
    
    return schedule


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup schedule wrapper."""
    
    def __init__(self, base_schedule, warmup_steps, target_lr):
        super().__init__()
        self.base_schedule = base_schedule
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
    
    def __call__(self, step):
        warmup_lr = self.target_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        
        if callable(self.base_schedule):
            base_lr = self.base_schedule(step - self.warmup_steps)
        else:
            base_lr = self.base_schedule
        
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: base_lr)
    
    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'target_lr': self.target_lr,
        }


def create_optimizer(args, learning_rate_schedule):
    """Creates optimizer.
    
    Args:
        args: Command line arguments.
        learning_rate_schedule: Learning rate schedule.
        
    Returns:
        Optimizer instance.
    """
    if args.optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    elif args.optimizer == 'adamw':
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_schedule,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'momentum':
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate_schedule,
            momentum=args.momentum)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')


def compute_loss(logits, labels, num_classes, ignore_label, 
                 top_k_percent_pixels=1.0):
    """Computes softmax cross entropy loss.
    
    Args:
        logits: Predicted logits [batch, height, width, num_classes].
        labels: Ground truth labels [batch, height, width, 1].
        num_classes: Number of classes.
        ignore_label: Label to ignore.
        top_k_percent_pixels: Fraction of pixels to use for loss.
        
    Returns:
        Scalar loss.
    """
    # Resize logits to match labels if needed
    label_shape = tf.shape(labels)
    logits = tf.image.resize(logits, [label_shape[1], label_shape[2]])
    
    # Flatten
    logits_flat = tf.reshape(logits, [-1, num_classes])
    labels_flat = tf.reshape(labels, [-1])
    
    # Create mask for valid pixels
    valid_mask = tf.not_equal(labels_flat, ignore_label)
    valid_indices = tf.where(valid_mask)
    
    # Filter to valid pixels
    valid_logits = tf.gather(logits_flat, valid_indices[:, 0])
    valid_labels = tf.gather(labels_flat, valid_indices[:, 0])
    
    # Compute per-pixel loss
    per_pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(valid_labels, tf.int32),
        logits=valid_logits)
    
    # Hard example mining (top-k percent pixels)
    if top_k_percent_pixels < 1.0:
        num_pixels = tf.shape(per_pixel_loss)[0]
        k = tf.cast(tf.cast(num_pixels, tf.float32) * top_k_percent_pixels, tf.int32)
        k = tf.maximum(k, 1)
        
        top_k_loss, _ = tf.math.top_k(per_pixel_loss, k=k, sorted=False)
        loss = tf.reduce_mean(top_k_loss)
    else:
        loss = tf.reduce_mean(per_pixel_loss)
    
    return loss


@tf.function
def train_step(model, optimizer, images, labels, num_classes, ignore_label,
               top_k_percent_pixels=1.0):
    """Performs one training step.
    
    Args:
        model: DeepLab model.
        optimizer: Optimizer.
        images: Input images.
        labels: Ground truth labels.
        num_classes: Number of classes.
        ignore_label: Label to ignore.
        top_k_percent_pixels: Fraction of pixels for hard mining.
        
    Returns:
        Loss value.
    """
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        logits = outputs[common.OUTPUT_TYPE]
        loss = compute_loss(logits, labels, num_classes, ignore_label,
                           top_k_percent_pixels)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    tf.get_logger().setLevel('INFO')
    
    # Create train directory
    os.makedirs(args.train_logdir, exist_ok=True)
    
    # Set up distribution strategy for multi-GPU
    if args.num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    else:
        strategy = tf.distribute.get_strategy()  # Default strategy
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create dataset
    with strategy.scope():
        dataset = wsi_data_generator.Dataset(
            dataset_name=args.dataset,
            dataset_dir=args.dataset_dir,
            batch_size=args.train_batch_size,
            crop_size=args.train_crop_size,
            downsample=args.wsi_downsample,
            augment_prob=args.augment_prob,
            min_scale_factor=args.min_scale_factor,
            max_scale_factor=args.max_scale_factor,
            scale_factor_step_size=args.scale_factor_step_size,
            model_variant=args.model_variant,
            num_readers=tf.data.AUTOTUNE,
            is_training=True,
            should_shuffle=True,
            should_repeat=True,
            ignore_label=args.ignore_label)
        
        train_dataset = dataset.get_dataset()
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    # Create model
    with strategy.scope():
        deeplab_model = DeepLabV3PlusModel(
            num_classes=args.num_classes,
            model_variant=args.model_variant,
            output_stride=args.output_stride,
            atrous_rates=tuple(args.atrous_rates) if args.atrous_rates else None,
            decoder_output_stride=tuple(args.decoder_output_stride) if args.decoder_output_stride else None,
            crop_size=(args.train_crop_size, args.train_crop_size),
            fine_tune_batch_norm=args.fine_tune_batch_norm)
        
        # Create learning rate schedule and optimizer
        lr_schedule = create_learning_rate_schedule(args, args.training_number_of_steps)
        optimizer = create_optimizer(args, lr_schedule)
        
        # Create checkpoint manager
        checkpoint = tf.train.Checkpoint(
            model=deeplab_model,
            optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.train_logdir,
            max_to_keep=5)
        
        # Load initial checkpoint if provided
        if args.tf_initial_checkpoint:
            print(f'Loading initial checkpoint from {args.tf_initial_checkpoint}')
            # Note: For TF1 checkpoints, use the tf1_to_tf2_mapper utility
            status = checkpoint.restore(args.tf_initial_checkpoint)
            status.expect_partial()
        
        # Restore from latest checkpoint if exists
        if checkpoint_manager.latest_checkpoint:
            print(f'Restoring from {checkpoint_manager.latest_checkpoint}')
            status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
            status.expect_partial()
    
    # Set up TensorBoard
    summary_writer = tf.summary.create_file_writer(args.train_logdir)
    
    # Training loop
    print(f'Starting training for {args.training_number_of_steps} steps')
    step = optimizer.iterations.numpy()
    
    train_iter = iter(train_dataset)
    
    while step < args.training_number_of_steps:
        start_time = time.time()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataset)
            batch = next(train_iter)
        
        images = batch[common.IMAGE]
        labels = batch[common.LABEL]
        
        # Compute top_k_percent for hard mining schedule
        if args.hard_example_mining_step > 0:
            progress = min(1.0, step / args.hard_example_mining_step)
            top_k = 1.0 - progress * (1.0 - args.top_k_percent_pixels)
        else:
            top_k = args.top_k_percent_pixels
        
        # Training step
        with strategy.scope():
            loss = train_step(
                deeplab_model, optimizer, images, labels,
                args.num_classes, args.ignore_label, top_k)
        
        step = optimizer.iterations.numpy()
        step_time = time.time() - start_time
        
        # Logging
        if step % args.log_steps == 0:
            lr = optimizer.learning_rate
            if callable(lr):
                lr = lr(step)
            print(f'Step {step}/{args.training_number_of_steps} - '
                  f'Loss: {loss:.4f}, LR: {lr:.6f}, '
                  f'Time: {step_time:.3f}s')
        
        # TensorBoard summaries
        if step % args.save_summaries_steps == 0:
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
                lr = optimizer.learning_rate
                if callable(lr):
                    lr = lr(step)
                tf.summary.scalar('learning_rate', lr, step=step)
        
        # Save checkpoint
        if step % args.save_interval_steps == 0:
            save_path = checkpoint_manager.save()
            print(f'Saved checkpoint: {save_path}')
    
    # Save final checkpoint
    save_path = checkpoint_manager.save()
    print(f'Training complete. Final checkpoint: {save_path}')
    
    # Save model in SavedModel format
    saved_model_path = os.path.join(args.train_logdir, 'saved_model')
    deeplab_model.save(saved_model_path)
    print(f'Saved model to: {saved_model_path}')


if __name__ == '__main__':
    main()
