#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=myeloma_eval_3D
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_dev
#SBATCH --time=04:00:00

cd /gpfs/home/kn2347/MARS
set -x -e

# Training setup
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE


module purge
module add anaconda3/gpu/2025.06
module load anaconda3/gpu/2025.06
module load gcc/11.2.0
module load qt5/5.11.1
conda activate myeloma6
module purge
module load slurm/current


export workdir='/gpfs/data/mayerhoeferlab/projects/myeloma/results/run_lars_on_mars_3D'

python /gpfs/home/kn2347/MARS/MARS/run_lars/code_3d/predictions.py \
    /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/run_lars_on_mars_3D/config.yaml \
    > ${workdir}/log_eval.log
