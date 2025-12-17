#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TensorFlow 1.x to TensorFlow 2.x Checkpoint Migration Utility.

TF2-MIGRATION: This module provides utilities to convert TF1 DeepLab checkpoints
to TF2 format. Key features:
- Lists all variables in a TF1 checkpoint
- Maps TF1 variable names to TF2 model weight names
- Handles name transformations (e.g., 'weights' -> 'kernel', 'biases' -> 'bias')
- Saves converted weights to TF2 checkpoint format
- Optionally exports to SavedModel or .weights.h5

Usage:
    python tf1_to_tf2_mapper.py \\
        --tf1_checkpoint=/path/to/model.ckpt \\
        --tf2_checkpoint=/path/to/output/tf2_checkpoint \\
        --model_variant=xception_65

Or use programmatically:
    from tf1_to_tf2_mapper import convert_tf1_to_tf2_checkpoint
    convert_tf1_to_tf2_checkpoint(tf1_path, tf2_path, model_variant)
"""

import os
import re
import argparse
import tensorflow as tf


# =============================================================================
# Variable Name Mapping Rules
# =============================================================================

# TF1 Slim -> TF2 Keras variable name mappings
TF1_TO_TF2_NAME_MAP = {
    # Convolutional layers
    'weights': 'kernel',
    'biases': 'bias',
    
    # Separable convolution
    'depthwise_weights': 'depthwise_kernel',
    'pointwise_weights': 'pointwise_kernel',
    
    # Batch normalization
    'gamma': 'gamma',
    'beta': 'beta',
    'moving_mean': 'moving_mean',
    'moving_variance': 'moving_variance',
}

# Scope name transformations
SCOPE_TRANSFORMS = [
    # Entry flow convolutions
    (r'xception_65/entry_flow/conv(\d+)_(\d+)/', r'entry_flow/conv\1_\2/'),
    
    # ASPP module
    (r'aspp(\d+)/', r'aspp_module/aspp\1/'),
    (r'image_pooling/', r'aspp_module/image_pooling/'),
    (r'concat_projection/', r'aspp_module/concat_projection/'),
    
    # Decoder
    (r'decoder/', r'decoder_module/'),
    
    # Logits
    (r'logits/', r'logits_layer/'),
]


def list_tf1_checkpoint_variables(checkpoint_path):
    """Lists all variables in a TF1 checkpoint.
    
    Args:
        checkpoint_path: Path to the TF1 checkpoint (without .index/.data suffix).
        
    Returns:
        List of tuples (variable_name, shape, dtype).
    """
    variables = []
    
    # Use TF1 compat mode to read checkpoint
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    
    for var_name in sorted(var_to_shape_map.keys()):
        shape = var_to_shape_map[var_name]
        dtype = var_to_dtype_map[var_name]
        variables.append((var_name, shape, dtype))
        
    return variables


def print_checkpoint_variables(checkpoint_path):
    """Prints all variables in a checkpoint for inspection.
    
    Args:
        checkpoint_path: Path to checkpoint.
    """
    print(f"\nVariables in checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    variables = list_tf1_checkpoint_variables(checkpoint_path)
    
    for var_name, shape, dtype in variables:
        print(f"  {var_name}")
        print(f"    Shape: {shape}, DType: {dtype}")
    
    print(f"\nTotal variables: {len(variables)}")


def transform_variable_name(tf1_name):
    """Transforms a TF1 variable name to TF2 format.
    
    Args:
        tf1_name: Original TF1 variable name.
        
    Returns:
        Transformed TF2 variable name.
    """
    tf2_name = tf1_name
    
    # Apply scope transformations
    for pattern, replacement in SCOPE_TRANSFORMS:
        tf2_name = re.sub(pattern, replacement, tf2_name)
    
    # Apply variable name mappings
    for tf1_suffix, tf2_suffix in TF1_TO_TF2_NAME_MAP.items():
        if tf2_name.endswith('/' + tf1_suffix):
            tf2_name = tf2_name[:-len(tf1_suffix)] + tf2_suffix
    
    # Remove ':0' suffix if present
    if tf2_name.endswith(':0'):
        tf2_name = tf2_name[:-2]
    
    return tf2_name


def create_variable_mapping(tf1_checkpoint_path, model=None, verbose=True):
    """Creates a mapping from TF1 checkpoint variables to TF2 model weights.
    
    Args:
        tf1_checkpoint_path: Path to TF1 checkpoint.
        model: Optional TF2 Keras model to verify weight names.
        verbose: Whether to print mapping details.
        
    Returns:
        Dictionary mapping TF1 variable names to TF2 weight names.
    """
    mapping = {}
    
    # Get TF1 variables
    tf1_variables = list_tf1_checkpoint_variables(tf1_checkpoint_path)
    
    # Get TF2 model weight names if model provided
    tf2_weight_names = set()
    if model is not None:
        for weight in model.weights:
            # Remove ':0' suffix
            name = weight.name.replace(':0', '')
            tf2_weight_names.add(name)
    
    # Create mapping
    unmapped = []
    for tf1_name, shape, dtype in tf1_variables:
        # Skip optimizer variables
        if 'Adam' in tf1_name or 'Momentum' in tf1_name or 'global_step' in tf1_name:
            continue
            
        tf2_name = transform_variable_name(tf1_name)
        
        if model is not None and tf2_name not in tf2_weight_names:
            # Try to find closest match
            candidates = [n for n in tf2_weight_names if 
                         tf2_name.split('/')[-1] in n]
            if candidates:
                tf2_name = candidates[0]
            else:
                unmapped.append((tf1_name, tf2_name))
                continue
        
        mapping[tf1_name] = tf2_name
    
    if verbose:
        print(f"\nVariable mapping created:")
        print(f"  Mapped: {len(mapping)}")
        print(f"  Unmapped: {len(unmapped)}")
        
        if unmapped:
            print("\nUnmapped variables:")
            for tf1_name, attempted_tf2_name in unmapped[:10]:
                print(f"  {tf1_name}")
                print(f"    -> (attempted) {attempted_tf2_name}")
            if len(unmapped) > 10:
                print(f"  ... and {len(unmapped) - 10} more")
    
    return mapping


def load_tf1_weights_to_dict(checkpoint_path, mapping=None):
    """Loads TF1 checkpoint weights into a dictionary.
    
    Args:
        checkpoint_path: Path to TF1 checkpoint.
        mapping: Optional name mapping dictionary.
        
    Returns:
        Dictionary mapping (transformed) variable names to numpy arrays.
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    weights = {}
    for var_name in var_to_shape_map.keys():
        # Skip optimizer variables
        if 'Adam' in var_name or 'Momentum' in var_name or 'global_step' in var_name:
            continue
        
        value = reader.get_tensor(var_name)
        
        if mapping is not None:
            if var_name in mapping:
                weights[mapping[var_name]] = value
        else:
            # Use transformed name
            tf2_name = transform_variable_name(var_name)
            weights[tf2_name] = value
    
    return weights


def assign_weights_to_model(model, weights_dict, verbose=True):
    """Assigns weights from dictionary to a Keras model.
    
    Args:
        model: TF2 Keras model.
        weights_dict: Dictionary mapping weight names to numpy arrays.
        verbose: Whether to print assignment details.
        
    Returns:
        Tuple of (num_assigned, num_missing, num_extra).
    """
    assigned = 0
    missing = []
    extra = list(weights_dict.keys())
    
    for weight in model.weights:
        weight_name = weight.name.replace(':0', '')
        
        # Try direct match
        if weight_name in weights_dict:
            weight.assign(weights_dict[weight_name])
            assigned += 1
            extra.remove(weight_name)
            continue
        
        # Try partial match (last component)
        weight_key = weight_name.split('/')[-1]
        matched = False
        for dict_name in list(weights_dict.keys()):
            if dict_name.endswith(weight_key) or weight_key in dict_name:
                # Check shape compatibility
                if weights_dict[dict_name].shape == tuple(weight.shape.as_list()):
                    weight.assign(weights_dict[dict_name])
                    assigned += 1
                    if dict_name in extra:
                        extra.remove(dict_name)
                    matched = True
                    break
        
        if not matched:
            missing.append(weight_name)
    
    if verbose:
        print(f"\nWeight assignment summary:")
        print(f"  Assigned: {assigned}/{len(model.weights)}")
        print(f"  Missing in checkpoint: {len(missing)}")
        print(f"  Extra in checkpoint: {len(extra)}")
        
        if missing:
            print("\nMissing weights (will use random initialization):")
            for name in missing[:10]:
                print(f"  {name}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
    
    return assigned, missing, extra


def convert_tf1_to_tf2_checkpoint(tf1_checkpoint_path,
                                   tf2_checkpoint_path,
                                   model=None,
                                   model_variant='xception_65',
                                   num_classes=None,
                                   output_stride=16,
                                   save_format='checkpoint'):
    """Converts a TF1 DeepLab checkpoint to TF2 format.
    
    Args:
        tf1_checkpoint_path: Path to TF1 checkpoint.
        tf2_checkpoint_path: Path for output TF2 checkpoint.
        model: Optional pre-built TF2 Keras model. If None, will attempt to build.
        model_variant: Model variant name (e.g., 'xception_65').
        num_classes: Number of output classes.
        output_stride: Output stride of the model.
        save_format: One of 'checkpoint', 'saved_model', or 'h5'.
        
    Returns:
        The model with loaded weights.
    """
    print(f"\nConverting TF1 checkpoint to TF2:")
    print(f"  Input: {tf1_checkpoint_path}")
    print(f"  Output: {tf2_checkpoint_path}")
    print(f"  Model: {model_variant}")
    
    # Load TF1 weights
    print("\nLoading TF1 weights...")
    weights_dict = load_tf1_weights_to_dict(tf1_checkpoint_path)
    print(f"  Loaded {len(weights_dict)} weight tensors")
    
    # If model not provided, we just save the raw weights
    if model is not None:
        # Assign weights to model
        print("\nAssigning weights to model...")
        assign_weights_to_model(model, weights_dict)
        
        # Save in requested format
        print(f"\nSaving in {save_format} format...")
        os.makedirs(os.path.dirname(tf2_checkpoint_path) or '.', exist_ok=True)
        
        if save_format == 'checkpoint':
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(tf2_checkpoint_path)
        elif save_format == 'saved_model':
            tf.saved_model.save(model, tf2_checkpoint_path)
        elif save_format == 'h5':
            model.save_weights(tf2_checkpoint_path + '.h5')
        else:
            raise ValueError(f"Unknown save_format: {save_format}")
            
        print(f"  Saved to: {tf2_checkpoint_path}")
        return model
    else:
        # Save weights dict as a raw checkpoint
        print("\nSaving raw weight tensors...")
        os.makedirs(os.path.dirname(tf2_checkpoint_path) or '.', exist_ok=True)
        
        # Create variables and checkpoint
        variables = {}
        for name, value in weights_dict.items():
            var_name = name.replace('/', '_').replace(':', '_')
            variables[var_name] = tf.Variable(value, name=name)
        
        checkpoint = tf.train.Checkpoint(**variables)
        checkpoint.save(tf2_checkpoint_path)
        print(f"  Saved {len(variables)} variables to: {tf2_checkpoint_path}")
        
        return weights_dict


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Convert TF1 DeepLab checkpoint to TF2 format')
    
    parser.add_argument('--tf1_checkpoint', type=str, required=True,
                        help='Path to TF1 checkpoint')
    parser.add_argument('--tf2_checkpoint', type=str, required=True,
                        help='Output path for TF2 checkpoint')
    parser.add_argument('--model_variant', type=str, default='xception_65',
                        help='Model variant (xception_65, mobilenet_v2, etc.)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of output classes')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='Output stride')
    parser.add_argument('--save_format', type=str, default='checkpoint',
                        choices=['checkpoint', 'saved_model', 'h5'],
                        help='Output format')
    parser.add_argument('--list_variables', action='store_true',
                        help='Just list variables in checkpoint')
    
    args = parser.parse_args()
    
    if args.list_variables:
        print_checkpoint_variables(args.tf1_checkpoint)
        return
    
    convert_tf1_to_tf2_checkpoint(
        tf1_checkpoint_path=args.tf1_checkpoint,
        tf2_checkpoint_path=args.tf2_checkpoint,
        model_variant=args.model_variant,
        num_classes=args.num_classes,
        output_stride=args.output_stride,
        save_format=args.save_format)


if __name__ == '__main__':
    main()
