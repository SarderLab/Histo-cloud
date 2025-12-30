#!/bin/bash
#SBATCH --job-name=HistoCloud
#SBATCH --partition=hpg-b200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --qos=pinaki.sarder
#SBATCH --account=pinaki.sarder
#SBATCH --output=HistoCloud_WSI_segmentation_%j.out
#SBATCH --error=HistoCloud_WSI_segmentation_%j.err

echo "Starting HistoCloud WSI segmentation job..."

# Load conda
module load conda
conda activate histo-cloud-tf2

# Set paths
CHECKPOINT="/home/iansari/model/model_mapped_tf2.ckpt"
DATA_DIR="/home/iansari/data/V10S14-085_XY03_21-0056.svs"
OUTPUT_DIR="/home/iansari/frozen_test4_tf1/test4_tf1/Histo-cloud/output"
OUTPUT_JSON="gloms-2.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd /home/iansari/frozen_test4_tf1/test4_tf1/Histo-cloud/histomicstk

# Set Python path to find deeplab modules
export PYTHONPATH=/home/iansari/frozen_test4_tf1/test4_tf1/Histo-cloud/histomicstk:$PYTHONPATH

# Run vis.py
python3 deeplab/vis.py \
    --model_variant xception_65 \
    --atrous_rates 6 \
    --atrous_rates 12 \
    --atrous_rates 18 \
    --output_stride 16 \
    --decoder_output_stride 4 \
    --save_json_annotation True \
    --checkpoint_dir "$CHECKPOINT" \
    --dataset_dir "$DATA_DIR" \
    --json_filename "$OUTPUT_DIR/$OUTPUT_JSON" \
    --vis_crop_size 2000 \
    --wsi_downsample 2 \
    --tile_step 1000 \
    --min_size 2000 \
    --vis_batch_size 1 \
    --vis_remove_border 100 \
    --simplify_contours 0.005 \
    --num_classes 2 \
    --class_names "gloms" \
    --save_heatmap False \
    --heatmap_stride 2 \
    --gpu 0

echo ""
echo "Annotation saved to: $OUTPUT_DIR/$OUTPUT_JSON"
