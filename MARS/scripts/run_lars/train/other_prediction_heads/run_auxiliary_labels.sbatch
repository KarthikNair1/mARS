#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=aux_labels
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short
#SBATCH --array=0-5
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

# Map array task IDs to auxiliary labels
declare -a labels=("seg_voxels_bin_25" "seg_voxels_bin_50" "seg_voxels_bin_75" "tumor_volume_bin_25" "tumor_volume_bin_50" "tumor_volume_bin_75")

label=${labels[${SLURM_ARRAY_TASK_ID}]}
output_dir="/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/other_prediction_heads/auxiliary_labels/${label}"

python /gpfs/home/kn2347/MARS/MARS/run_lars/train.py /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/train/other_prediction_heads/auxiliary_labels_config.yaml \
    --auxiliary_head_label_names ${label} \
    --output ${output_dir} \
    --WANDB_run_name ${label} \
    --seed_index 0 --checkpoint /gpfs/data/mayerhoeferlab/projects/lars/lars-avg-max_chpt/LARS_trained_state-dict_0.pth
