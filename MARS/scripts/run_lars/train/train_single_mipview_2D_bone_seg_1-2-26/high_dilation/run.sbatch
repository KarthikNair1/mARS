#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=mm_dl
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short
#SBATCH --time=3-00:00:00
#SBATCH --exclude=a100-4005,a100-4020

cd /gpfs/home/kn2347/MARS
set -x -e

module add anaconda3/gpu/2025.06
module load anaconda3/gpu/2025.06
module load gcc/11.2.0
module load qt5/5.11.1
conda activate myeloma6
module purge
module load slurm/current

export WANDB_API_KEY='e51b9830c285698aced24214131f58ee2bfbd5e3'
export WANDB_CACHE_DIR='/gpfs/home/kn2347/.cache/wandb'
export WANDB_CONFIG_DIR='/gpfs/home/kn2347/.config/wandb'
export WANDB_DIR='/gpfs/home/kn2347/wandb'

idx=${SLURM_ARRAY_TASK_ID}
python MARS/run_lars/train.py /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/train/train_single_mipview_2D_bone_seg_1-2-26/high_dilation/training_config.yaml