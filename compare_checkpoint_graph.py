"""Compare TF1 checkpoint variables with graph variables."""
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
import numpy as np
from collections import defaultdict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'histomicstk'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'histomicstk', 'deeplab'))

from deeplab import common
from deeplab import model

# Parse flags
if not hasattr(tf.compat.v1.flags.FLAGS, '__parsed'):
    tf.compat.v1.flags.FLAGS(sys.argv)


def load_checkpoint_variables(checkpoint_path):
    """Load and return all variables from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    try:
        # Get checkpoint reader
        reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
        
        # Get variable to shape map
        var_to_shape_map = reader.get_variable_to_shape_map()
        
        checkpoint_vars = {}
        for var_name in sorted(var_to_shape_map.keys()):
            var_shape = var_to_shape_map[var_name]
            var_dtype = reader.get_tensor(var_name).dtype
            checkpoint_vars[var_name] = {
                'shape': var_shape,
                'dtype': var_dtype,
                'num_params': np.prod(var_shape) if var_shape else 0
            }
        
        return checkpoint_vars
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def build_graph_variables():
    """Build graph and return all variables."""
    print("\nBuilding graph...")
    
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: 2},
        crop_size=[512, 512],
        atrous_rates=[6, 12, 18],
        output_stride=16
    )
    
    with tf.Graph().as_default() as graph:
        input_image = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[1, 512, 512, 3], 
            name='input_image'
        )
        
        try:
            predictions = model.predict_labels(
                input_image,
                model_options=model_options,
                image_pyramid=None
            )
        except Exception as e:
            print(f"Error building graph: {e}")
            return None, None
        
        all_vars = tf.compat.v1.global_variables()
        
        graph_vars = {}
        for var in all_vars:
            graph_vars[var.name] = {
                'shape': var.shape.as_list(),
                'dtype': var.dtype.name,
                'num_params': np.prod(var.shape.as_list()) if var.shape.as_list() else 0,
                'trainable': var in tf.compat.v1.trainable_variables()
            }
        
        return graph_vars, graph


def compare_variables(checkpoint_vars, graph_vars):
    """Compare checkpoint and graph variables."""
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    # Normalize names for comparison (remove :0 suffix from graph var names)
    ckpt_names = set(checkpoint_vars.keys())
    graph_names_raw = set(graph_vars.keys())
    graph_names_normalized = {name.rstrip(':0') for name in graph_names_raw}
    
    # Create mapping from normalized to original graph names
    norm_to_orig = {name.rstrip(':0'): name for name in graph_names_raw}
    
    print(f"Checkpoint variables: {len(ckpt_names)}")
    print(f"Graph variables: {len(graph_names_raw)}")
    
    # Calculate total parameters
    ckpt_total_params = sum(v['num_params'] for v in checkpoint_vars.values())
    graph_total_params = sum(v['num_params'] for v in graph_vars.values())
    
    print(f"\nCheckpoint total parameters: {ckpt_total_params:,}")
    print(f"Graph total parameters: {graph_total_params:,}")
    
    # Find common variables
    common_vars = ckpt_names & graph_names_normalized
    print(f"\nCommon variables (in both): {len(common_vars)}")
    
    # Variables only in checkpoint
    only_in_ckpt = ckpt_names - graph_names_normalized
    print(f"Only in checkpoint: {len(only_in_ckpt)}")
    
    # Variables only in graph
    only_in_graph = graph_names_normalized - ckpt_names
    print(f"Only in graph: {len(only_in_graph)}")
    
    # Analyze common variables
    print("\n" + "="*80)
    print("COMMON VARIABLES ANALYSIS")
    print("="*80 + "\n")
    
    shape_mismatches = []
    dtype_mismatches = []
    
    for var_name in sorted(common_vars):
        ckpt_info = checkpoint_vars[var_name]
        graph_info = graph_vars[norm_to_orig[var_name]]
        
        # Check shape match
        ckpt_shape = list(ckpt_info['shape'])
        graph_shape = graph_info['shape']
        
        if ckpt_shape != graph_shape:
            shape_mismatches.append({
                'name': var_name,
                'ckpt_shape': ckpt_shape,
                'graph_shape': graph_shape
            })
        
        # Check dtype match
        if str(ckpt_info['dtype']) != graph_info['dtype']:
            dtype_mismatches.append({
                'name': var_name,
                'ckpt_dtype': ckpt_info['dtype'],
                'graph_dtype': graph_info['dtype']
            })
    
    print(f"Variables with matching shapes and dtypes: {len(common_vars) - len(shape_mismatches)}")
    print(f"Variables with shape mismatches: {len(shape_mismatches)}")
    print(f"Variables with dtype mismatches: {len(dtype_mismatches)}")
    
    # Show shape mismatches
    if shape_mismatches:
        print("\n" + "-"*80)
        print("SHAPE MISMATCHES:")
        print("-"*80)
        for mismatch in shape_mismatches[:20]:  # Show first 20
            print(f"\n  {mismatch['name']}")
            print(f"    Checkpoint: {mismatch['ckpt_shape']}")
            print(f"    Graph:      {mismatch['graph_shape']}")
        if len(shape_mismatches) > 20:
            print(f"\n  ... and {len(shape_mismatches) - 20} more")
    
    # Analyze variables only in checkpoint
    if only_in_ckpt:
        print("\n" + "="*80)
        print("VARIABLES ONLY IN CHECKPOINT (not in graph)")
        print("="*80)
        print("\nThese are typically:")
        print("  - Optimizer state variables (e.g., Adam/Momentum accumulators)")
        print("  - Global step counter")
        print("  - Moving averages")
        print("  - Training-only variables\n")
        
        # Group by type
        by_category = defaultdict(list)
        for var_name in sorted(only_in_ckpt):
            if 'global_step' in var_name.lower():
                by_category['Global Step'].append(var_name)
            elif any(x in var_name.lower() for x in ['adam', 'momentum', 'rmsprop', 'optimizer']):
                by_category['Optimizer State'].append(var_name)
            elif 'moving' in var_name.lower():
                by_category['Moving Averages'].append(var_name)
            elif 'beta' in var_name or 'gamma' in var_name:
                by_category['BatchNorm Parameters'].append(var_name)
            else:
                by_category['Other'].append(var_name)
        
        for category, vars_list in sorted(by_category.items()):
            print(f"\n{category} ({len(vars_list)} variables):")
            for var_name in vars_list[:10]:  # Show first 10 of each category
                info = checkpoint_vars[var_name]
                print(f"  - {var_name}")
                print(f"    Shape: {info['shape']}, Dtype: {info['dtype']}")
            if len(vars_list) > 10:
                print(f"  ... and {len(vars_list) - 10} more")
    
    # Analyze variables only in graph
    if only_in_graph:
        print("\n" + "="*80)
        print("VARIABLES ONLY IN GRAPH (not in checkpoint)")
        print("="*80)
        print("\nThese could be:")
        print("  - Newly added layers/modules")
        print("  - Architecture changes")
        print("  - Variables with renamed scopes\n")
        
        for var_name in sorted(only_in_graph)[:30]:  # Show first 30
            orig_name = norm_to_orig[var_name]
            info = graph_vars[orig_name]
            trainable_str = "trainable" if info['trainable'] else "non-trainable"
            print(f"  - {var_name}")
            print(f"    Shape: {info['shape']}, Dtype: {info['dtype']}, {trainable_str}")
        
        if len(only_in_graph) > 30:
            print(f"\n  ... and {len(only_in_graph) - 30} more")
    
    # Summary of restoration potential
    print("\n" + "="*80)
    print("CHECKPOINT RESTORATION ANALYSIS")
    print("="*80 + "\n")
    
    restorable = len(common_vars) - len(shape_mismatches)
    graph_trainable = sum(1 for v in graph_vars.values() if v.get('trainable', False))
    
    print(f"Graph has {len(graph_names_raw)} variables ({graph_trainable} trainable)")
    print(f"Checkpoint has {len(ckpt_names)} variables")
    print(f"\nRestorable variables: {restorable} / {len(graph_names_raw)} ({100*restorable/len(graph_names_raw):.1f}%)")
    
    if only_in_graph:
        print(f"\n⚠ WARNING: {len(only_in_graph)} graph variables missing from checkpoint!")
        print("  These will be randomly initialized if you load the checkpoint.")
    
    if shape_mismatches:
        print(f"\n⚠ WARNING: {len(shape_mismatches)} variables have shape mismatches!")
        print("  These cannot be restored and will cause errors.")
    
    if only_in_ckpt:
        print(f"\n✓ INFO: {len(only_in_ckpt)} checkpoint variables not used in graph.")
        print("  This is normal (optimizer state, moving averages, etc.)")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only
    
    checkpoint_path = '/home/iansari/model/model.ckpt-5000'
    
    print("="*80)
    print("TF1 CHECKPOINT VS GRAPH VARIABLE COMPARISON")
    print("="*80)
    
    # Load checkpoint variables
    checkpoint_vars = load_checkpoint_variables(checkpoint_path)
    if checkpoint_vars is None:
        print("\n✗ Failed to load checkpoint. Exiting.")
        return
    
    print(f"✓ Loaded {len(checkpoint_vars)} variables from checkpoint")
    
    # Build graph and get variables
    graph_vars, graph = build_graph_variables()
    if graph_vars is None:
        print("\n✗ Failed to build graph. Exiting.")
        return
    
    print(f"✓ Built graph with {len(graph_vars)} variables")
    
    # Compare
    compare_variables(checkpoint_vars, graph_vars)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
