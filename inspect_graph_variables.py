"""Script to initialize DeepLab graph and inspect variables."""
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'histomicstk'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'histomicstk', 'deeplab'))

from deeplab import common
from deeplab import model

# Parse flags to avoid UnparsedFlagAccessError
import sys as _sys
if not hasattr(tf.compat.v1.flags.FLAGS, '__parsed'):
    tf.compat.v1.flags.FLAGS(_sys.argv)

def inspect_graph_variables():
    """Initialize graph and print variable information."""
    print("\n" + "="*80)
    print("INITIALIZING DEEPLAB GRAPH")
    print("="*80 + "\n")
    
    # Set up basic model options
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: 2},  # Binary segmentation
        crop_size=[512, 512],  # Standard crop size
        atrous_rates=[6, 12, 18],  # Standard rates for output_stride=16
        output_stride=16
    )
    
    print(f"Model Options:")
    print(f"  - Crop size: {model_options.crop_size}")
    print(f"  - Output stride: {model_options.output_stride}")
    print(f"  - Atrous rates: {model_options.atrous_rates}")
    print(f"  - Number of classes: {model_options.outputs_to_num_classes}\n")
    
    # Build the graph
    with tf.Graph().as_default() as graph:
        # Create a dummy input image
        input_image = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[1, 512, 512, 3], 
            name='input_image'
        )
        
        print("Building model graph...")
        try:
            # Build the model
            predictions = model.predict_labels(
                input_image,
                model_options=model_options,
                image_pyramid=None
            )
            print("✓ Model graph built successfully\n")
        except Exception as e:
            print(f"✗ Error building model: {e}\n")
            return
        
        # Get all variables
        all_vars = tf.compat.v1.global_variables()
        
        print("="*80)
        print(f"GRAPH VARIABLES SUMMARY")
        print("="*80 + "\n")
        print(f"Total number of variables: {len(all_vars)}\n")
        
        # Calculate total parameters
        total_params = 0
        trainable_params = 0
        
        # Group variables by scope
        var_by_scope = {}
        for var in all_vars:
            scope = var.name.split('/')[0]
            if scope not in var_by_scope:
                var_by_scope[scope] = []
            var_by_scope[scope].append(var)
            
            # Count parameters
            var_params = np.prod(var.shape.as_list())
            total_params += var_params
            if var in tf.compat.v1.trainable_variables():
                trainable_params += var_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
        
        # Print variables by scope
        print("="*80)
        print("VARIABLES BY SCOPE")
        print("="*80 + "\n")
        
        for scope, vars_in_scope in sorted(var_by_scope.items()):
            scope_params = sum(np.prod(v.shape.as_list()) for v in vars_in_scope)
            print(f"\n{scope}/ ({len(vars_in_scope)} variables, {scope_params:,} parameters)")
            print("-" * 80)
        
        # Print detailed variable information
        print("\n" + "="*80)
        print("DETAILED VARIABLE INFORMATION")
        print("="*80 + "\n")
        
        for i, var in enumerate(all_vars, 1):
            shape = var.shape.as_list()
            dtype = var.dtype.name
            num_params = np.prod(shape) if shape else 0
            trainable = "✓" if var in tf.compat.v1.trainable_variables() else "✗"
            
            print(f"{i:4d}. {var.name}")
            print(f"       Shape: {shape}")
            print(f"       Dtype: {dtype}")
            print(f"       Parameters: {num_params:,}")
            print(f"       Trainable: {trainable}")
            print()
        
        # Print graph operations summary
        print("\n" + "="*80)
        print("GRAPH OPERATIONS SUMMARY")
        print("="*80 + "\n")
        
        ops = graph.get_operations()
        print(f"Total operations in graph: {len(ops)}\n")
        
        # Count operations by type
        op_types = {}
        for op in ops:
            op_type = op.type
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        print("Top 20 operation types:")
        for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {op_type:30s}: {count:5d}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")


if __name__ == '__main__':
    # Disable GPU for inspection (CPU only)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Run inspection
    inspect_graph_variables()
