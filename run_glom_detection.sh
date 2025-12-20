#!/bin/bash
# Run vis.py with TF2 mapped checkpoint to generate glomeruli annotations

# Activate conda environment
module load conda
conda activate histo-cloud-tf2

# Set paths
CHECKPOINT="/home/iansari/model/model_mapped_tf2.ckpt"
DATASET_DIR="$1"  # Pass WSI path as first argument
OUTPUT_JSON="${2:-gloms.json}"  # Output JSON name (default: gloms.json)

# Check if dataset_dir provided
if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <WSI_PATH> [OUTPUT_JSON_NAME]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/slide.svs gloms.json"
    exit 1
fi

# Run vis.py
cd /home/iansari/test4_tf1/Histo-cloud/histomicstk/deeplab

python vis.py \
    --checkpoint_dir="$CHECKPOINT" \
    --dataset_dir="$DATASET_DIR" \
    --dataset='wsi_dataset' \
    --model_variant='xception_65' \
    --output_stride=16 \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --vis_crop_size=512 \
    --vis_batch_size=1 \
    --wsi_downsample=1 \
    --tile_step=256 \
    --vis_remove_border=50 \
    --num_classes=2 \
    --class_names='background,glomerulus' \
    --min_size=100 \
    --simplify_contours=0.001 \
    --save_json_annotation=True \
    --json_filename="$OUTPUT_JSON" \
    --gpu='0'

echo ""
echo "Annotation saved to: $OUTPUT_JSON"
