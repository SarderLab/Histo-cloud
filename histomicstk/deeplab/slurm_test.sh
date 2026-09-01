#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7000mb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce
#SBATCH --time=72:00:00
#SBATCH --output=./slurm_log_test.out
#SBATCH --job-name="segmentation_frozen"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load singularity
ls
ml

# Add your userid here:
USER=sayat.mimar
# Add the name of the folder containing WSIs here
PROJECT=segmentation

CODESDIR=/blue/pinaki.sarder/sayat.mimar/segmentation_test/Histo-cloud/histomicstk/deeplab

DATADIR=/$CODESDIR/test_data
MODELDIR=$CODESDIR/log_dir

CONTAINER=$CODESDIR/myhistorepo_histo_img.sif

singularity exec --nv -B $(pwd):/exec/,$DATADIR/:/data $CONTAINER python3 /exec/vis.py --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --checkpoint_dir $MODELDIR/model.ckpt-5000 --dataset_dir /data/ --vis_crop_size 2000 --wsi_downsample 2 --tile_step 1000 --min_size 2000 --vis_batch_size 1 --vis_remove_border 100 --simplify_contours 0.005 --num_classes 2 --class_names 'gloms' --save_heatmap=False --heatmap_stride 2 --gpu 0
