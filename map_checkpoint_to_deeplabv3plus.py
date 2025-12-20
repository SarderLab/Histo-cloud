"""
Map TF1 DeepLabV3+ checkpoint to TF2 graph with Xception-65 backbone.
Based on the strategy from past_mapper.py.
"""
import argparse
import os
import sys
import datetime
import json
import pathlib
import tempfile
import shutil
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tensorflow as tf
import numpy as np

# Add paths FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'histomicstk'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'histomicstk', 'deeplab'))

# Import deeplab modules
from deeplab import common
from deeplab import model

# Now parse flags and override model_variant and decoder settings
tf.compat.v1.flags.FLAGS([sys.argv[0]])  # Parse with just script name
tf.compat.v1.flags.FLAGS.model_variant = 'xception_65'  # Override to use Xception-65
tf.compat.v1.flags.FLAGS.decoder_output_stride = ['4']  # Enable decoder for Xception-65


def load_tf1_tensors(ckpt_prefix: str):
    """Return dict: {name: (ndarray, shape)} for all tensors in a TF1 checkpoint."""
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt_prefix)
    ckpt_var_to_shape = reader.get_variable_to_shape_map()
    tensors = {}
    for k, shape in ckpt_var_to_shape.items():
        tensors[k] = (reader.get_tensor(k), tuple(shape))
    return tensors


def build_deeplabv3plus_graph(num_classes=2, crop_size=512, output_stride=16):
    """
    Build DeepLabV3+ graph with Xception-65 backbone.
    Returns graph and all variables.
    """
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: num_classes},
        crop_size=[crop_size, crop_size],
        atrous_rates=[6, 12, 18],
        output_stride=output_stride
    )
    
    graph = tf.Graph()
    with graph.as_default():
        input_image = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[1, crop_size, crop_size, 3], 
            name='input_image'
        )
        
        # Build model with Xception-65
        predictions = model.predict_labels(
            input_image,
            model_options=model_options,
            image_pyramid=None
        )
        
        # Create global_step variable (needed for inference)
        global_step = tf.compat.v1.train.get_or_create_global_step()
        
        all_vars = tf.compat.v1.global_variables()
        
    return graph, all_vars


def get_graph_variables(var_list):
    """
    Returns list of (name, shape_tuple, var) for all TF variables, sorted by name.
    """
    out = []
    for v in sorted(var_list, key=lambda x: x.name):
        out.append((v.name, tuple(v.shape.as_list()), v))
    return out


def _strip_tf_suffixes(name: str) -> str:
    """Remove TF tensor suffix :0"""
    if name.endswith(':0'):
        name = name[:-2]
    return name


def should_skip_tf1_var(name: str) -> bool:
    """
    Skip optimizer slot variables (but KEEP BatchNorm moving averages for inference).
    """
    skip_patterns = [
        '/Momentum',
        '/Adam',
        '/RMSProp',
        'global_step'
    ]
    for pattern in skip_patterns:
        if pattern in name:
            return True
    return False


def normalize_tf1_name(name: str) -> str:
    """
    Normalize TF1 variable names to match TF2 graph structure.
    
    Common patterns in DeepLabV3+ checkpoint:
    - xception_65/entry_flow/...
    - xception_65/middle_flow/...
    - xception_65/exit_flow/...
    - decoder/...
    - aspp0/weights
    - aspp1_depthwise/depthwise_weights
    - logits/semantic/...
    
    TF2 graph uses tf.compat.v1.variable_scope which maintains similar names.
    """
    name = _strip_tf_suffixes(name)
    
    # Most DeepLabV3+ variables should match directly after normalization
    # The main differences are:
    # 1. BatchNorm beta/gamma/moving_mean/moving_variance (we skip moving stats)
    # 2. Optimizer state (we skip these)
    
    return name


def normalize_tf2_name(name: str) -> str:
    """
    Canonicalize TF2 variable names by stripping ':0' and removing duplicate scopes.
    
    TF2 creates duplicate scopes for BatchNorm like:
      xception_65/.../separable_conv1_depthwise/BatchNorm/xception_65/.../separable_conv1_depthwise/BatchNorm/beta:0
    Which should map to TF1:
      xception_65/.../separable_conv1_depthwise/BatchNorm/beta
    
    The pattern is: prefix/BatchNorm/prefix/BatchNorm/param -> prefix/BatchNorm/param
    """
    name = _strip_tf_suffixes(name)
    
    # Check if this is a BatchNorm variable with duplicate scope
    if '/BatchNorm/' in name:
        # Find all occurrences of /BatchNorm/
        bn_indices = []
        idx = 0
        while True:
            idx = name.find('/BatchNorm/', idx)
            if idx == -1:
                break
            bn_indices.append(idx)
            idx += 1
        
        # If we have exactly 2 occurrences, check for duplication
        if len(bn_indices) == 2:
            first_bn = bn_indices[0]
            second_bn = bn_indices[1]
            
            # Extract parts
            prefix = name[:first_bn]  # Everything before first /BatchNorm/
            middle = name[first_bn+11:second_bn]  # Between the two /BatchNorm/
            suffix = name[second_bn+11:]  # After second /BatchNorm/
            
            # Check if middle == prefix
            if middle == prefix:
                # Remove duplication
                return f"{prefix}/BatchNorm/{suffix}"
    
    return name


def build_normalized_maps(tf1_ckpt_map, tf2_var_list):
    """
    Build normalized dicts for TF1 and TF2:
      tf1_norm: {norm_name: (np_array, shape_tuple, original_name)}
      tf2_norm: {norm_name: (shape_tuple, var_obj, original_name)}
    """
    tf1_norm = {}
    for k, (arr, shape) in tf1_ckpt_map.items():
        if should_skip_tf1_var(k):
            continue
        norm = normalize_tf1_name(k)
        if norm in tf1_norm:
            print(f"  WARNING: Duplicate normalized TF1 name: {norm}")
        tf1_norm[norm] = (arr, shape, k)

    tf2_norm = {}
    for name, shape, var in tf2_var_list:
        norm = normalize_tf2_name(name)
        if norm in tf2_norm:
            print(f"  WARNING: Duplicate normalized TF2 name: {norm}")
        tf2_norm[norm] = (shape, var, name)

    return tf1_norm, tf2_norm


def match_and_assign(tf1_norm, tf2_norm, graph, strict_shape=True, assign=True):
    """
    Attempt to match by normalized name and (optionally) shape.
    Returns diagnostics and creates assignment ops in the graph.
    """
    matched = []
    tf1_only = []
    tf2_only = []
    assign_ops = []

    # First pass: exact name matches
    for n, (shape2, var2, orig2) in tf2_norm.items():
        if n in tf1_norm:
            arr1, shape1, orig1 = tf1_norm[n]
            shapes_ok = (tuple(shape1) == tuple(shape2))
            if not shapes_ok:
                if strict_shape:
                    tf2_only.append((orig2, shape2, "shape_mismatch_with_tf1", shape1))
                    continue
                else:
                    print(f"  WARNING: Shape mismatch but continuing: {orig2}")
                    print(f"           TF1: {shape1}, TF2: {shape2}")
            
            # Create assignment operation in the graph
            if assign and shapes_ok:
                with graph.as_default():
                    assign_op = var2.assign(arr1)
                    assign_ops.append(assign_op)
            
            matched.append((orig1, shape1, orig2, shape2))
        else:
            tf2_only.append((orig2, shape2, "no_tf1_match", None))

    # TF1 leftovers
    tf2_norm_names = set(tf2_norm.keys())
    for n, (arr1, shape1, orig1) in tf1_norm.items():
        if n not in tf2_norm_names:
            tf1_only.append((orig1, shape1, "no_tf2_match"))

    return matched, tf1_only, tf2_only, assign_ops


def pretty_report(matched, tf1_only, tf2_only, max_list=40):
    def _fmt_pairs(pairs, n=10):
        return "\n".join([f"  - {p}" for p in pairs[:n]]) + ("" if len(pairs) <= n else f"\n  ... and {len(pairs)-n} more")

    print("\n" + "="*80)
    print("MAPPING REPORT")
    print("="*80)
    print(f"\nMatched assignments: {len(matched)}")
    if matched:
        print(_fmt_pairs([(m[0] + "  ->  " + m[2]) for m in matched], max_list))

    print(f"\n\nTF1-only (no TF2 match): {len(tf1_only)}")
    print("These are typically optimizer state, moving averages, or training-only vars.")
    if tf1_only:
        print(_fmt_pairs([f"{n}  {s}  ({why})" for (n, s, why) in tf1_only], max_list))

    print(f"\n\nTF2-only (no TF1 match OR shape mismatch): {len(tf2_only)}")
    print("⚠ WARNING: These will be randomly initialized!")
    if tf2_only:
        def fmt(t):
            name2, shape2, why, extra = t
            if why == "shape_mismatch_with_tf1":
                return f"{name2}  {shape2}  (shape mismatch vs TF1 {extra})"
            return f"{name2}  {shape2}  ({why})"
        print(_fmt_pairs([fmt(t) for t in tf2_only], max_list))
    
    print("\n" + "="*80)
    
    # Summary statistics
    total_tf2 = len(matched) + len(tf2_only)
    coverage = 100 * len(matched) / total_tf2 if total_tf2 > 0 else 0
    print(f"\nRESTORATION SUMMARY:")
    print(f"  Total TF2 variables: {total_tf2}")
    print(f"  Successfully matched: {len(matched)} ({coverage:.1f}%)")
    print(f"  Missing from checkpoint: {len(tf2_only)}")
    print("="*80 + "\n")


def save_checkpoint(graph, output_path, assign_ops):
    """
    Execute assignment ops and save a new TF2-compatible checkpoint.
    """
    with tf.compat.v1.Session(graph=graph) as sess:
        # Initialize all variables first
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Execute all assignments
        print(f"\nExecuting {len(assign_ops)} assignment operations...")
        sess.run(assign_ops)
        print("✓ All assignments completed")
        
        # Save checkpoint
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, output_path)
        print(f"✓ Checkpoint saved to: {save_path}")
        
        return save_path


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    parser = argparse.ArgumentParser(
        description="Map TF1 DeepLabV3+ checkpoint to TF2 graph with Xception-65"
    )
    parser.add_argument(
        "--ckpt", 
        default="/home/iansari/model/model.ckpt-5000",
        help="TF1 checkpoint prefix (no .index/.data suffix)"
    )
    parser.add_argument(
        "--output", 
        default=None,
        help="Output path for mapped checkpoint (default: input_path + '_mapped')"
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=2,
        help="Number of output classes"
    )
    parser.add_argument(
        "--crop_size", 
        type=int, 
        default=512,
        help="Input crop size"
    )
    parser.add_argument(
        "--output_stride", 
        type=int, 
        default=16,
        help="Output stride (8 or 16)"
    )
    parser.add_argument(
        "--no_assign", 
        action="store_true", 
        help="Dry-run: do not assign weights or save"
    )
    parser.add_argument(
        "--non_strict_shapes", 
        action="store_true", 
        help="Allow shape mismatch (will NOT assign mismatched vars)"
    )
    args = parser.parse_args()

    # Disable GPU for checkpoint operations
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("="*80)
    print("TF1 TO TF2 CHECKPOINT MAPPER - DeepLabV3+ with Xception-65")
    print("="*80 + "\n")

    print("[1/6] Reading TF1 checkpoint...")
    tf1_tensors = load_tf1_tensors(args.ckpt)
    print(f"     ✓ Loaded {len(tf1_tensors)} tensors from TF1 checkpoint")

    print("\n[2/6] Building TF2 graph with Xception-65...")
    graph, all_vars = build_deeplabv3plus_graph(
        num_classes=args.num_classes,
        crop_size=args.crop_size,
        output_stride=args.output_stride
    )
    print(f"     ✓ Built graph")

    print("\n[3/6] Extracting variables from TF2 graph...")
    tf2_vars = get_graph_variables(all_vars)
    print(f"     ✓ Found {len(tf2_vars)} variables in TF2 graph")

    print("\n[4/6] Normalizing names...")
    tf1_norm, tf2_norm = build_normalized_maps(tf1_tensors, tf2_vars)
    print(f"      TF1 normalized: {len(tf1_norm)}  |  TF2 normalized: {len(tf2_norm)}")

    print("\n[5/6] Matching and creating assignment ops...")
    matched, tf1_only, tf2_only, assign_ops = match_and_assign(
        tf1_norm,
        tf2_norm,
        graph,
        strict_shape=not args.non_strict_shapes,
        assign=not args.no_assign
    )

    pretty_report(matched, tf1_only, tf2_only)

    if not args.no_assign and assign_ops:
        print("\n[6/6] Saving mapped checkpoint...")
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_path = args.ckpt.rsplit('.ckpt', 1)[0]
            output_path = f"{base_path}_mapped_tf2.ckpt"
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        save_checkpoint(graph, output_path, assign_ops)
        
        # Save metadata
        metadata = {
            "source_checkpoint": args.ckpt,
            "mapped_at": _timestamp(),
            "num_classes": args.num_classes,
            "crop_size": args.crop_size,
            "output_stride": args.output_stride,
            "matched_variables": len(matched),
            "total_tf2_variables": len(tf2_vars),
            "coverage_percent": 100 * len(matched) / len(tf2_vars) if tf2_vars else 0,
            "tensorflow_version": tf.__version__
        }
        
        metadata_path = output_path + ".metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        print("\n" + "="*80)
        print("CHECKPOINT MAPPING COMPLETE!")
        print("="*80)
        print(f"\nMapped checkpoint: {output_path}")
        print(f"Metadata: {metadata_path}")
        print(f"\nCoverage: {metadata['coverage_percent']:.1f}%")
        
        if len(tf2_only) > 0:
            print(f"\n⚠ WARNING: {len(tf2_only)} variables will be randomly initialized!")
            print("  Review the mapping report above for details.")
    
    elif args.no_assign:
        print("\n[6/6] Skipped (--no_assign flag)")
        print("\nDry-run complete. Use without --no_assign to save the mapped checkpoint.")


if __name__ == "__main__":
    main()
