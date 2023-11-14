#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7000mb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce
#SBATCH --time=72:00:00
#SBATCH --output=./slurm_log.out
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

DATADIR=/$CODESDIR/training_data 
MODELDIR=$CODESDIR/trained_model

CONTAINER=$CODESDIR/myhistorepo_histo_img.sif

singularity exec --nv -B $(pwd):/exec/,$DATADIR/:/data,$MODELDIR/:/model/ $CONTAINER python3 /exec/train.py --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --train_crop_size 400 --train_batch_size 2 --training_number_of_steps 5000 --slow_start_step 1000 --augment_prob 0.01 --slow_start_learning_rate 1e-05 --base_learning_rate 0.0005 --tf_initial_checkpoint /model/model.ckpt-400000 --dataset_dir /data/ --train_logdir $CODESDIR/log_dir --save_interval_secs 600 --num_clones 1 --global_step 0 --end_learning_rate 0.0 --learning_power 0.9 --ignore_label 2 --decay_steps 0 --last_layer_gradient_multiplier 10.0 --wsi_downsample 1 --wsi_downsample 2 --wsi_downsample 3 --wsi_downsample 4 --initialize_last_layer=false --fine_tune_batch_norm=false --last_layers_contain_logits_only=false --upsample_logits=true

