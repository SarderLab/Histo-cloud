#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_logs_%j.out
#SBATCH --job-name="HistoCloud-training"

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anish.tatke@ufl.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load singularity
pwd
ls
ml

# Add your userid here:
USER=anish.tatke
# Add the name of the folder containing WSIs here
PROJECT=HistoCloud-training

BLUEDIR=/blue/pinaki.sarder/anish.tatke/$PROJECT
ORANGEDIR=/orange/pinaki.sarder/anish.tatke/$PROJECT

DATADIR=/$ORANGEDIR/LNR01_Test
MODELDIR=$ORANGEDIR/model_ckpt

CONTAINER=$BLUEDIR/container/myhistorepo_histo_img.sif

singularity exec --writable $CONTAINER pip3 install large-image[all] --find-links https://girder.github.io/large_image_wheels

singularity exec --nv -B $(pwd):/exec/,$DATADIR/:/data $CONTAINER python3 /exec/train_network.py \
    --girderApiUrl 'https://devathena.rc.ufl.edu/api/v1' \
    --girderToken 0EzHs4NNN66IOK84Otbgr3bhTpQ4D5jTgO0JJKXOAPpJ7xXDonNcvef9JHG0qjsw \
    --inputFolder /data \
    --inputFolderID "67900d7958b173229fd609cc" \
    --inputModelFile $MODELDIR/model-Glomeruli-11-13-20.zip \
    --outputModel $MODELDIR/outputModel.zip \
    --classes "non_globally_sclerotic_glomeruli" \
    --patch_size 400 \
    --batch_size 2 \
    --steps 5000 \
    --WSI_downsample 1 2 3 4 \
    --learning_rate 0.0005 \
    --learning_rate_start 0.00001 \
    --slow_start_step 1000 \
    --init_last_layer="false" \
    --batch_norm="false" \
    --augment 0.01 \
    --num_clones 1 \
    --global_step 0 \
    --end_learning_rate 0.0 \
    --learning_power 0.9 \
    --ignore_label="ignore" \
    --decay_steps 0 \
    --last_layer_gradient_multiplier 10 \
    --last_layers_contai n_logits_only="false" \
    --upsample_logits="true" \
    --gpu 1