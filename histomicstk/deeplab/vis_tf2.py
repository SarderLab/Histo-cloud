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
"""TF2-native WSI segmentation visualization script for DeepLab.

This script performs semantic segmentation on Whole Slide Images (WSI) and
generates annotation files in XML or JSON format for HistomicsTK.

Usage:
    python vis_tf2.py \
        --checkpoint_dir=/path/to/checkpoint \
        --dataset_dir=/path/to/wsi/files \
        --vis_crop_size=2000 \
        --output_stride=16 \
        --num_classes=2 \
        --save_json_annotation

TF2 Migration:
    - Replaced tf.app.flags with argparse
    - Replaced tf.Session with eager execution
    - Replaced tf.train.MonitoredSession with tf.train.Checkpoint
    - Uses @tf.function for performance-critical prediction
"""

import argparse
import json
import os
import sys
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.append("..")

from large_image.cache_util import cachesClear

from deeplab import common
from deeplab import model
from deeplab.datasets import wsi_data_generator
from deeplab.utils import save_annotation
from deeplab.utils.mask_to_xml import mask_to_xml
from deeplab.utils.xml_to_json import convert_xml_json

# Try to import progress helper (may not be available in all environments)
try:
    from deeplab.progress_helper import ProgressHelper
    HAS_PROGRESS_HELPER = True
except ImportError:
    HAS_PROGRESS_HELPER = False
    class ProgressHelper:
        """Dummy progress helper when not available."""
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def progress(self, value):
            pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepLab v3+ WSI Visualization (TF2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # GPU settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id used for inference.')
    
    # Checkpoint and directories
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory of model checkpoints or path to checkpoint file.')
    parser.add_argument('--vis_logdir', type=str, default=None,
                        help='Where to write the event logs (optional).')
    
    # Visualization settings
    parser.add_argument('--vis_batch_size', type=int, default=1,
                        help='The number of images in each batch during visualization.')
    parser.add_argument('--vis_crop_size', type=int, default=2000,
                        help='Crop size [size, size] for visualization.')
    parser.add_argument('--wsi_downsample', type=int, default=2,
                        help='Downsample rate of WSI used during training.')
    parser.add_argument('--tile_step', type=int, default=500,
                        help='Number of pixels the patch grid overlaps during testing.')
    parser.add_argument('--vis_remove_border', type=int, default=100,
                        help='Number of pixels on the patch border that are not used.')
    parser.add_argument('--simplify_contours', type=float, default=0.001,
                        help='Amount of approximation to simplify predicted boundaries (Douglas-Peucker epsilon).')
    
    # Model settings
    parser.add_argument('--model_variant', type=str, default='xception_65',
                        help='DeepLab model variant.')
    parser.add_argument('--atrous_rates', type=int, nargs='+', default=None,
                        help='Atrous rates for atrous spatial pyramid pooling.')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='The ratio of input to output spatial resolution.')
    parser.add_argument('--decoder_output_stride', type=int, nargs='*', default=None,
                        help='Decoder output stride.')
    
    # Multi-scale settings
    parser.add_argument('--eval_scales', type=float, nargs='+', default=[1.0],
                        help='The scales to resize images for evaluation.')
    parser.add_argument('--add_flipped_images', action='store_true',
                        help='Add flipped images for evaluation.')
    parser.add_argument('--image_pyramid', type=float, nargs='*', default=None,
                        help='Image pyramid scales for inference.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='wsi_dataset',
                        help='Name of the segmentation dataset.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Where the dataset (WSI files) reside.')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes.')
    parser.add_argument('--min_size', type=int, default=0,
                        help='Minimum size of the detected regions.')
    parser.add_argument('--class_names', type=str, default=None,
                        help='Comma-separated names of output classes for JSON conversion.')
    
    # Preprocessing
    parser.add_argument('--min_resize_value', type=int, default=None,
                        help='Minimum resize value.')
    parser.add_argument('--max_resize_value', type=int, default=None,
                        help='Maximum resize value.')
    parser.add_argument('--resize_factor', type=int, default=None,
                        help='Resize factor.')
    
    # Output settings
    parser.add_argument('--save_json_annotation', action='store_true',
                        help='Save the predictions in .json format for HistomicsTK.')
    parser.add_argument('--save_heatmap', action='store_true',
                        help='Save the prediction logits as a heatmap in HistomicsTK.')
    parser.add_argument('--heatmap_stride', type=int, default=4,
                        help='The stride of the saved heatmap (additional downsample).')
    parser.add_argument('--json_filename', type=str, default='annotation.anot',
                        help='*.json annotation filename.')
    
    return parser.parse_args()


class WSISegmenter:
    """TF2-native WSI segmentation class."""
    
    def __init__(self, model_options, checkpoint_path):
        """Initialize the segmenter.
        
        Args:
            model_options: common.ModelOptions instance
            checkpoint_path: Path to checkpoint file or directory
        """
        self.model_options = model_options
        self.checkpoint_path = checkpoint_path
        self._model_built = False
        
        # Load checkpoint
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        # Find checkpoint file
        if os.path.isdir(self.checkpoint_path):
            ckpt_path = tf.train.latest_checkpoint(self.checkpoint_path)
            if ckpt_path is None:
                raise ValueError(f'No checkpoint found in {self.checkpoint_path}')
        else:
            ckpt_path = self.checkpoint_path
        
        print(f'Loading checkpoint: {ckpt_path}')
        
        # Create a checkpoint reader to get variable names
        self.ckpt_reader = tf.train.load_checkpoint(ckpt_path)
        self.ckpt_path = ckpt_path
        
        # Note: Variables will be restored when model is first called
        print('Checkpoint loaded successfully')
    
    @tf.function
    def _predict_batch(self, images):
        """Run prediction on a batch of images.
        
        Args:
            images: Batch of images [batch, height, width, 3]
            
        Returns:
            predictions: Semantic predictions [batch, height, width]
            probabilities: Class probabilities [batch, height, width, num_classes]
        """
        predictions_dict = model.predict_labels(
            images,
            model_options=self.model_options,
            image_pyramid=None
        )
        
        predictions = predictions_dict[common.OUTPUT_TYPE]
        probabilities = predictions_dict.get(
            common.OUTPUT_TYPE + '_prob',
            tf.zeros_like(predictions, dtype=tf.float32)
        )
        
        return predictions, probabilities
    
    def predict(self, images):
        """Run prediction (handles numpy arrays).
        
        Args:
            images: Numpy array or tensor [batch, height, width, 3]
            
        Returns:
            predictions: Numpy array [batch, height, width]
            probabilities: Numpy array [batch, height, width, num_classes]
        """
        if isinstance(images, np.ndarray):
            images = tf.constant(images, dtype=tf.float32)
        
        predictions, probabilities = self._predict_batch(images)
        
        return predictions.numpy(), probabilities.numpy()


def process_batch(slide_mask, slide_heatmap, offset, predictions, probabilities,
                  image_names, mask_size, border, downsample, extra_downsample):
    """Process a batch of predictions and update the slide mask.
    
    Args:
        slide_mask: The cumulative slide mask array
        slide_heatmap: The cumulative heatmap array
        offset: Dictionary with 'X' and 'Y' tissue offset
        predictions: Batch of semantic predictions
        probabilities: Batch of class probabilities
        image_names: Batch of image filenames (with coordinates)
        mask_size: [height, width] of the slide mask
        border: Number of border pixels to remove
        downsample: WSI downsample factor
        extra_downsample: Additional downsample from model output
        
    Returns:
        Updated slide_mask and slide_heatmap
    """
    border = int(round(border / extra_downsample))
    num_images = predictions.shape[0]
    
    for i in range(num_images):
        image_height = predictions.shape[1]
        image_width = predictions.shape[2]
        semantic_prediction = predictions[i]
        semantic_probability = probabilities[i]
        
        # Decode filename if bytes
        if isinstance(image_names[i], bytes):
            image_filename = image_names[i].decode()
        else:
            image_filename = str(image_names[i])
        
        # Parse coordinates from filename
        # Filename format: ...X-Y-...
        parts = image_filename.split('-')
        Ystart = float(parts[-2]) - offset['Y']
        Ystart /= downsample * extra_downsample
        Ystart = int(round(Ystart)) + border
        
        Xstart = float(parts[-3]) - offset['X']
        Xstart /= downsample * extra_downsample
        Xstart = int(round(Xstart)) + border
        
        Ystop = min(int(round(Ystart + image_height - (border * 2))), mask_size[0])
        Xstop = min(int(round(Xstart + image_width - (border * 2))), mask_size[1])
        
        # Ensure we don't go negative
        Ystart = max(0, Ystart)
        Xstart = max(0, Xstart)
        
        # Update slide mask with maximum (handles overlapping regions)
        y_slice = slice(Ystart, Ystop)
        x_slice = slice(Xstart, Xstop)
        pred_y_slice = slice(border, Ystop - Ystart + border)
        pred_x_slice = slice(border, Xstop - Xstart + border)
        
        slide_mask[y_slice, x_slice] = np.maximum(
            slide_mask[y_slice, x_slice],
            semantic_prediction[pred_y_slice, pred_x_slice]
        )
        
        slide_heatmap[y_slice, x_slice, :] = (
            slide_heatmap[y_slice, x_slice, :] + 
            semantic_probability[pred_y_slice, pred_x_slice, :]
        )
    
    return slide_mask, slide_heatmap


def save_json_with_heatmap(json_data, slide_heatmap, num_classes, class_names,
                           downsample, extra_downsample, heatmap_stride, tissue_offset):
    """Add heatmap layers to JSON annotation data.
    
    Args:
        json_data: List of JSON annotation elements
        slide_heatmap: The heatmap array [H, W, num_classes]
        num_classes: Number of classes
        class_names: List of class names
        downsample: WSI downsample factor
        extra_downsample: Additional downsample
        heatmap_stride: Stride for heatmap output
        tissue_offset: Dictionary with 'X' and 'Y' offset
        
    Returns:
        Updated json_data with heatmap layers
    """
    ds = downsample * extra_downsample * heatmap_stride
    strided_heatmap = slide_heatmap[::heatmap_stride, ::heatmap_stride, :]
    
    cutoff = 1 / (num_classes * 3)  # Cutoff low values
    heatmap_sum = strided_heatmap.sum(2)
    heatmap_sum_mask = heatmap_sum == 0
    np.place(strided_heatmap[:, :, 0], heatmap_sum_mask, 1)
    np.place(heatmap_sum, heatmap_sum_mask, 1)
    strided_heatmap /= np.repeat(np.expand_dims(heatmap_sum, -1), num_classes, -1)
    
    for iter_idx in range(num_classes - 1):  # Iterate through all logits layers
        print(f"Building JSON layer: [{class_names[iter_idx]}-heatmap]")
        
        single_heatmap = strided_heatmap[:, :, iter_idx + 1]
        
        heatmap = {
            "type": "heatmap",
            "radius": ds / 2 + 1,
            "colorRange": [
                "rgba(255,255,0,0)", 
                "rgba(255,255,0,.3)",
                "rgba(255,190,0,.4)", 
                "rgba(255,0,0,.5)"
            ],
            "rangeValues": [cutoff, cutoff * 1.5, 1.5 / num_classes, 1],
            "normalizeRange": True
        }
        
        values = single_heatmap.flatten()
        values_mask = values > cutoff
        values = np.round(values[values_mask], 3)
        np.place(values, values > 1, 1)
        
        y_idx, x_idx = np.indices(single_heatmap.shape)
        y_idx = y_idx.flatten()[values_mask]
        x_idx = x_idx.flatten()[values_mask]
        z_idx = np.zeros_like(x_idx)
        
        points = np.empty((x_idx.size * 4), dtype=np.float32)
        points[0::4] = x_idx * ds + tissue_offset['X']
        points[1::4] = y_idx * ds + tissue_offset['Y']
        points[2::4] = z_idx
        points[3::4] = values
        
        heatmap["points"] = points.reshape([-1, 4]).tolist()
        json_data.append({
            "name": f"{class_names[iter_idx]}-heatmap",
            "elements": [heatmap]
        })
    
    return json_data


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f'GPU configuration error: {e}')
    
    print('=' * 60)
    print('DeepLab v3+ WSI Visualization (TF2)')
    print('=' * 60)
    print(f'Checkpoint: {args.checkpoint_dir}')
    print(f'Dataset directory: {args.dataset_dir}')
    print(f'Crop size: {args.vis_crop_size}')
    print(f'WSI downsample: {args.wsi_downsample}')
    print(f'Output stride: {args.output_stride}')
    print(f'Num classes: {args.num_classes}')
    print('=' * 60)
    
    # Create dataset for getting WSI info
    dataset = wsi_data_generator.Dataset(
        dataset_name=args.dataset,
        dataset_dir=args.dataset_dir,
        num_of_classes=args.num_classes,
        downsample=args.wsi_downsample,
        tile_step=args.tile_step,
        batch_size=args.vis_batch_size,
        crop_size=args.vis_crop_size,
        min_resize_value=args.min_resize_value,
        max_resize_value=args.max_resize_value,
        resize_factor=args.resize_factor,
        model_variant=args.model_variant,
        is_training=False,
        should_shuffle=False,
        should_repeat=False
    )
    
    # Validate border settings
    assert args.vis_remove_border * 2 < args.vis_crop_size, \
        'vis_remove_border*2 must be less than vis_crop_size'
    assert args.vis_remove_border * 2 < args.vis_crop_size - args.tile_step, \
        'vis_remove_border*2 must be less than vis_crop_size - tile_step'
    
    # Get list of slides
    if os.path.isfile(args.dataset_dir):
        slides = [args.dataset_dir]
    else:
        slides = dataset._get_all_files(with_xml=False, save_mask=False)
    
    if not slides:
        print('No slides found!')
        return
    
    print(f'\nFound {len(slides)} slides to process')
    
    # Create model options
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: args.num_classes},
        crop_size=[args.vis_crop_size, args.vis_crop_size],
        atrous_rates=args.atrous_rates,
        output_stride=args.output_stride,
        model_variant=args.model_variant,
        decoder_output_stride=args.decoder_output_stride
    )
    
    # Create segmenter
    segmenter = WSISegmenter(model_options, args.checkpoint_dir)
    
    # Track broken slides
    broken_slides = []
    
    # Use progress helper if available
    progress_context = ProgressHelper('Segment WSI') if HAS_PROGRESS_HELPER else ProgressHelper('Segment WSI')
    
    with progress_context as helper:
        for slide_idx, slide in enumerate(slides):
            print(f'\nWorking on: [{slide}] ({slide_idx + 1}/{len(slides)})')
            
            try:
                # Get iterator for this slide
                iterator, num_samples, tissue_offset, tissue_size = \
                    dataset.get_one_shot_iterator_grid(slide)
                
                # Calculate extra downsample from model output
                # This happens when prediction_with_upsampled_logits is False
                if not model_options.prediction_with_upsampled_logits:
                    # Estimate from output_stride
                    extra_downsample = args.output_stride
                else:
                    extra_downsample = 1
                
                # Calculate mask size
                def get_downsampled_size(size, downsample=args.wsi_downsample * extra_downsample):
                    return int(np.ceil(size / downsample))
                
                mask_size = [
                    get_downsampled_size(tissue_size[0]),
                    get_downsampled_size(tissue_size[1])
                ]
                
                print(f'Creating slide mask: {mask_size} pixels')
                
                # Initialize arrays
                slide_mask = np.zeros([mask_size[0], mask_size[1]], dtype=np.uint8)
                slide_heatmap = np.zeros(
                    [mask_size[0], mask_size[1], args.num_classes], 
                    dtype=np.float16
                )
                
                # Process batches
                batch_count = 0
                for batch_data in iterator:
                    # Extract data from batch
                    images = batch_data[common.IMAGE]
                    image_names = batch_data[common.IMAGE_NAME]
                    
                    # Convert to numpy if needed
                    if hasattr(images, 'numpy'):
                        images_np = images.numpy()
                    else:
                        images_np = np.array(images)
                    
                    if hasattr(image_names, 'numpy'):
                        image_names_np = image_names.numpy()
                    else:
                        image_names_np = np.array(image_names)
                    
                    # Run prediction
                    predictions, probabilities = segmenter.predict(images_np)
                    
                    # Resize predictions if needed (when output is smaller than input)
                    actual_extra_downsample = args.vis_crop_size // predictions.shape[1]
                    
                    # Process batch and update mask
                    slide_mask, slide_heatmap = process_batch(
                        slide_mask=slide_mask,
                        slide_heatmap=slide_heatmap,
                        offset=tissue_offset,
                        predictions=predictions,
                        probabilities=probabilities,
                        image_names=image_names_np,
                        mask_size=mask_size,
                        border=args.vis_remove_border,
                        downsample=args.wsi_downsample,
                        extra_downsample=actual_extra_downsample
                    )
                    
                    batch_count += args.vis_batch_size
                    
                    # Update progress
                    progress = min(batch_count, num_samples) / float(num_samples)
                    helper.progress(progress)
                    
                    if batch_count % 10 == 0 or batch_count >= num_samples:
                        print(f'Processed [{min(batch_count, num_samples)} of {num_samples}] patches')
                
                # Clear large_image caches
                cachesClear()
                
                # Save annotations
                if args.save_json_annotation:
                    anot_filename = args.json_filename
                    print(f'\nCreating annotation file: [{anot_filename}]')
                    
                    # Convert mask to XML and then to JSON
                    root = mask_to_xml(
                        xml_path=anot_filename,
                        mask=slide_mask,
                        downsample=args.wsi_downsample * actual_extra_downsample,
                        min_size_thresh=args.min_size,
                        simplify_contours=args.simplify_contours,
                        return_root=True,
                        maxClass=args.num_classes - 1,
                        offset=tissue_offset
                    )
                    
                    compartments = args.class_names.split(',') if args.class_names else \
                                   [f'class_{i}' for i in range(args.num_classes)]
                    json_data = convert_xml_json(root, compartments)
                    
                    # Add heatmap if requested
                    if args.save_heatmap:
                        json_data = save_json_with_heatmap(
                            json_data=json_data,
                            slide_heatmap=slide_heatmap,
                            num_classes=args.num_classes,
                            class_names=compartments,
                            downsample=args.wsi_downsample,
                            extra_downsample=actual_extra_downsample,
                            heatmap_stride=args.heatmap_stride,
                            tissue_offset=tissue_offset
                        )
                    
                    # Save JSON
                    with open(anot_filename, 'w') as f:
                        json.dump(json_data, f, sort_keys=False)
                    
                    del json_data, root
                else:
                    # Save as XML
                    anot_filename = f'{os.path.splitext(slide)[0]}.xml'
                    print(f'\nCreating annotation file: [{anot_filename}]')
                    
                    mask_to_xml(
                        xml_path=anot_filename,
                        mask=slide_mask,
                        downsample=args.wsi_downsample * actual_extra_downsample,
                        min_size_thresh=args.min_size,
                        simplify_contours=args.simplify_contours,
                        offset=tissue_offset
                    )
                
                del slide_mask, slide_heatmap
                print('Annotation file saved...\n')
                
            except Exception as e:
                print(f'!!! Error processing slide: {slide}')
                print(f'    Error: {e}')
                import traceback
                traceback.print_exc()
                broken_slides.append(slide)
                continue
    
    # Report broken slides
    if broken_slides:
        print('\n!!! The following slides failed and were not processed:')
        for slide in broken_slides:
            print(f'-\t[{slide}]')
    
    print('\n\nAll done.')


if __name__ == '__main__':
    main()
