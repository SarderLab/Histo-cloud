# TensorFlow 1.x → TensorFlow 2.x Migration Plan

## Overview

This document outlines the migration strategy for converting the DeepLab v3+ codebase from TensorFlow 1.x (with TF-Slim and tf.contrib) to native TensorFlow 2.x using tf.keras APIs.

**Migration Status: COMPLETE** ✅

---

## Migration Summary

### Completed Files

| File | Status | Description |
|------|--------|-------------|
| `core/utils.py` | ✅ Complete | SplitSeparableConv2D Keras layer, updated resize ops |
| `core/preprocess_utils.py` | ✅ Complete | Updated deprecated ops (tf.random.uniform, etc.) |
| `core/xception.py` | ✅ Complete | SeparableConv2DSame, XceptionModule Keras layers |
| `core/feature_extractor.py` | ✅ Complete | NetworkConfig class, removed arg_scope |
| `model.py` | ✅ Complete | ASPPModule, DecoderModule, updated all functions |
| `common.py` | ✅ Complete | Removed tf.app.flags, DEFAULTS class |
| `datasets/data_generator.py` | ✅ Complete | tf.io API updates, get_dataset() method |
| `tf1_to_tf2_mapper.py` | ✅ New | Checkpoint conversion utility |
| `train_tf2.py` | ✅ New | TF2-native training script |
| `eval_tf2.py` | ✅ New | TF2-native evaluation script |
| `export_model_tf2.py` | ✅ New | SavedModel/TFLite export |

---

## Migration Goals

1. **Target**: Fully TF2-native implementation of DeepLab v3+:
   - Eager execution by default with `@tf.function` on performance-critical paths
   - `tf.keras.Model` subclasses and `tf.keras.layers` instead of TF-Slim
   - Modern `tf.data.Dataset` pipelines for input
   - Object-based checkpoints with `tf.train.Checkpoint`
   - No TF-Slim, no tf.contrib, no graph/session training loops

2. **Constraints**:
   - Preserve architecture (ASPP, decoder, backbone choices)
   - Enable one-time TF1→TF2 weight mapping
   - Minimize changes while achieving TF2 nativeness

---

## Key API Mappings

| TF1 Pattern | TF2 Replacement |
|-------------|-----------------|
| `tf.contrib.slim` | `tf.keras.layers` |
| `tf.contrib.layers.batch_norm` | `tf.keras.layers.BatchNormalization` |
| `slim.arg_scope` | Layer configuration in `__init__` |
| `tf.variable_scope` | `tf.keras.layers` automatic naming |
| `tf.get_variable` | `tf.Variable` or layer weights |
| `tf.train.Saver` | `tf.train.Checkpoint` |
| `tf.Session` | Eager execution / `@tf.function` |
| `tf.app.flags` | `argparse` or `absl.flags` |
| `tf.train.MomentumOptimizer` | `tf.keras.optimizers.SGD(momentum=)` |
| `tf.train.AdamOptimizer` | `tf.keras.optimizers.Adam` |
| `tf.contrib.metrics` | `tf.keras.metrics` |
| `slim.separable_conv2d` | `tf.keras.layers.SeparableConv2D` |
| `slim.conv2d` | `tf.keras.layers.Conv2D` |
| `slim.avg_pool2d` | `tf.keras.layers.AveragePooling2D` |

---

## Migration Order (Dependency-Based)

### Phase 1: Core Utilities ✅ COMPLETE
Files to migrate first as they have minimal dependencies:

1. **`core/utils.py`** ✅ MIGRATED
   - Created `SplitSeparableConv2D` Keras layer
   - Replaced `slim.batch_norm` references
   - Updated `resize_bilinear` to `tf.image.resize`
   - Replaced `tf.to_float` → `tf.cast(..., tf.float32)`

2. **`core/preprocess_utils.py`** ✅ MIGRATED
   - `tf.random_uniform` → `tf.random.uniform`
   - `tf.lin_space` → `tf.linspace`
   - `tf.to_int32` → `tf.cast(..., tf.int32)`
   - `tf.image.resize_bilinear` → `tf.image.resize`

3. **`common.py`** ✅ MIGRATED
   - Replaced `tf.app.flags` → `_DefaultConfig` class
   - `tf.gfile.Open` → `tf.io.gfile.GFile`
   - `ModelOptions` now accepts all config via kwargs
   - Added backwards compatibility `FLAGS` shim

### Phase 2: Backbone Networks ✅ COMPLETE

4. **`core/xception.py`** ✅ MIGRATED
   - Created `SeparableConv2DSame` Keras layer (handles SAME padding with dilation)
   - Created `XceptionModule` Keras layer for blocks
   - Removed `slim.separable_conv2d`, `slim.conv2d`, `slim.batch_norm`
   - Removed `slim.arg_scope` dependency
   - Added `training` parameter support

5. **`core/resnet_v1_beta.py`** - TBD (not used in current workflow)
6. **`nets/mobilenet/*.py`** - TBD (not used in current workflow)

### Phase 3: Feature Extractor ✅ COMPLETE

7. **`core/feature_extractor.py`** ✅ MIGRATED
   - Created `NetworkConfig` class (replaces arg_scope pattern)
   - Created `config_map` (replaces `arg_scopes_map`)
   - Removed `slim.arg_scope` entirely
   - Updated `extract_features()` with explicit config passing
   - Removed deprecated `reuse` parameter

### Phase 4: DeepLab Model Components ✅ COMPLETE

8. **`model.py`** ✅ MIGRATED (Major refactor ~900 lines)
   - Created `SplitSeparableConv2D` Keras layer
   - Created `ASPPModule` Keras layer (multi-branch pooling)
   - Created `DecoderModule` Keras layer
   - Updated `extract_features()`, `refine_by_decoder()`
   - Updated `_decoder_with_sum_merge()`, `_decoder_with_concat_merge()`
   - Updated `get_branch_logits()`, `multi_scale_logits()`
   - Removed all `slim.arg_scope` patterns

### Phase 5: Data Pipeline ✅ COMPLETE

9. **`datasets/data_generator.py`** ✅ MIGRATED
   - `tf.FixedLenFeature` → `tf.io.FixedLenFeature`
   - `tf.parse_single_example` → `tf.io.parse_single_example`
   - `tf.gfile.Glob` → `tf.io.gfile.glob`
   - Added `get_dataset()` method for TF2 access

10. **`datasets/wsi_data_generator.py`** - TBD (similar pattern)
11. **`input_preprocess.py`** - Minimal changes needed

### Phase 6: Training Infrastructure ✅ COMPLETE

12. **`train_tf2.py`** ✅ NEW FILE CREATED
   - `DeepLabV3PlusModel` Keras Model class
   - `WarmupSchedule` learning rate warmup
   - `create_learning_rate_schedule()` (poly/cosine/step)
   - `create_optimizer()` with Adam/SGD support
   - `compute_loss()` with hard example mining
   - `@tf.function` decorated `train_step()`
   - Full `tf.distribute.Strategy` support (MirroredStrategy, TPU)
   - TensorBoard integration
   - Checkpoint management
   - ~450 lines of TF2-native training code

### Phase 7: Evaluation & Inference ✅ COMPLETE

13. **`eval_tf2.py`** ✅ NEW FILE CREATED
   - `MeanIoU` Keras metric with per-class IoU
   - `DeepLabV3PlusEvaluator` class
   - Multi-scale evaluation support
   - Continuous checkpoint watching
   - TensorBoard logging
   - ~400 lines of TF2-native evaluation code

14. **`export_model_tf2.py`** ✅ NEW FILE CREATED
   - `DeepLabExportModel` Keras Model wrapper
   - SavedModel export with signatures
   - TFLite export with optional quantization
   - Supports both uint8 input and preprocessed float32 input
   - ~400 lines

### Phase 8: Checkpoint Migration ✅ COMPLETE

15. **`tf1_to_tf2_mapper.py`** ✅ NEW FILE CREATED
   - `list_tf1_checkpoint_variables()` - enumerate TF1 vars
   - `transform_variable_name()` - name mapping (weights→kernel, etc.)
   - `create_variable_mapping()` - build mapping dict
   - `load_tf1_weights_to_dict()` - load TF1 weights
   - `assign_weights_to_model()` - assign to Keras model
   - `convert_tf1_to_tf2_checkpoint()` - end-to-end conversion

---

## Component Architecture (TF2)

```
DeepLabV3Plus (tf.keras.Model)
├── backbone (XceptionBackbone / ResNetBackbone / MobileNetBackbone)
│   └── Returns: features, end_points dict
├── aspp_module (ASPPModule - tf.keras.layers.Layer)
│   ├── image_pooling branch
│   ├── 1x1 conv branch
│   ├── atrous conv branches (rates: 6, 12, 18 or 12, 24, 36)
│   └── concat + projection
├── decoder (DecoderModule - tf.keras.layers.Layer)
│   ├── feature_projection (1x1 conv on low-level features)
│   ├── upsample and concat
│   └── refinement convolutions
└── logits_layer (tf.keras.layers.Conv2D)
```

---

## Variable Name Mapping Strategy

To enable TF1→TF2 checkpoint loading:

1. **Preserve scope names** where possible:
   - `xception_65/entry_flow/...` → same in Keras model
   - `aspp/...` → `aspp_module/...`
   - `decoder/...` → `decoder_module/...`

2. **Handle BatchNorm variables**:
   - TF1: `gamma`, `beta`, `moving_mean`, `moving_variance`
   - TF2 Keras: `gamma`, `beta`, `moving_mean`, `moving_variance`
   - Names should align naturally

3. **Handle Conv2D variables**:
   - TF1 slim: `weights`, `biases`
   - TF2 Keras: `kernel`, `bias`
   - Mapper must handle this transformation

4. **Handle SeparableConv2D**:
   - TF1: `depthwise_weights`, `pointwise_weights`
   - TF2: `depthwise_kernel`, `pointwise_kernel`

---

## Testing Strategy

1. **Unit Tests**: Verify each component produces same output shapes
2. **Integration Tests**: Load TF1 weights, verify predictions match
3. **Numerical Tests**: Compare intermediate activations between TF1 and TF2

---

## Files NOT Requiring Migration

- `deeplab_demo.ipynb` - Can be updated separately
- Test files (`*_test.py`) - Update after main migration
- `slim/` directory - Will be replaced, not migrated

---

## Notes on Behavioral Changes

1. **BatchNorm behavior**: TF2 Keras BatchNorm handles `training` flag automatically when using `model.fit()`. For custom training loops, explicitly pass `training=True/False`.

2. **Variable initialization**: TF2 initializes eagerly. No need for `tf.global_variables_initializer()`.

3. **Control dependencies**: Most explicit `tf.control_dependencies` are unnecessary in eager mode.

4. **Name scopes vs variable scopes**: TF2 uses name scopes for organization; Keras layers handle variable naming automatically.

---

## Migration Markers

Throughout the migration, we use these comment markers:

```python
# TF2-MIGRATION: [description of change]
# TF2-MIGRATION-NOTE: [behavioral note for future reference]
# TF2-MIGRATION-TODO: [remaining work needed]
```

---

## Usage Guide (TF2)

### Training

```bash
python train_tf2.py \
    --dataset_dir=/path/to/dataset \
    --train_logdir=/path/to/logs \
    --dataset=pascal_voc_seg \
    --train_split=train \
    --model_variant=xception_65 \
    --atrous_rates 6 12 18 \
    --output_stride=16 \
    --decoder_output_stride 4 \
    --train_crop_size 513 513 \
    --train_batch_size=4 \
    --training_number_of_steps=30000 \
    --base_learning_rate=0.007 \
    --learning_rate_policy=poly
```

### Evaluation

```bash
python eval_tf2.py \
    --checkpoint_dir=/path/to/checkpoints \
    --eval_logdir=/path/to/eval_logs \
    --dataset_dir=/path/to/dataset \
    --dataset=pascal_voc_seg \
    --eval_split=val \
    --eval_crop_size 513 513 \
    --model_variant=xception_65
```

### Export Model

```bash
# Export as SavedModel
python export_model_tf2.py \
    --checkpoint_path=/path/to/checkpoint \
    --export_path=/path/to/export \
    --export_format=saved_model \
    --num_classes=21 \
    --crop_size 513 513

# Export as TFLite
python export_model_tf2.py \
    --checkpoint_path=/path/to/checkpoint \
    --export_path=/path/to/model.tflite \
    --export_format=tflite \
    --num_classes=21 \
    --tflite_quantize
```

### Convert TF1 Checkpoint to TF2

```python
from deeplab import tf1_to_tf2_mapper

# List TF1 variables
tf1_to_tf2_mapper.list_tf1_checkpoint_variables('/path/to/tf1/checkpoint')

# Full conversion (requires building TF2 model first)
tf1_to_tf2_mapper.convert_tf1_to_tf2_checkpoint(
    tf1_checkpoint_path='/path/to/tf1/checkpoint',
    tf2_model=my_keras_model,
    output_path='/path/to/tf2/checkpoint'
)
```

### Programmatic Usage

```python
from deeplab import common, model

# Create model options
model_options = common.ModelOptions(
    outputs_to_num_classes={common.OUTPUT_TYPE: 21},
    crop_size=[513, 513],
    atrous_rates=[6, 12, 18],
    output_stride=16,
    model_variant='xception_65',
    decoder_output_stride=[4]
)

# Run inference
predictions = model.predict_labels(
    images,  # [batch, height, width, 3] float32
    model_options=model_options,
    image_pyramid=None
)

semantic_pred = predictions[common.OUTPUT_TYPE]  # [batch, height, width]
```
