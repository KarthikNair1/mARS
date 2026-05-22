#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=tumor50_sweep_agent
#SBATCH --mem=100GB
#SBATCH --partition=a100_short
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/other_prediction_heads/auxiliary_labels/tumor_volume_bin_50_sweep-%A_%a.out
#SBATCH --error=/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/other_prediction_heads/auxiliary_labels/tumor_volume_bin_50_sweep-%A_%a.err

cd /gpfs/home/kn2347/MARS
set -x -e

module add anaconda3/gpu/2025.06
module load anaconda3/gpu/2025.06
module load gcc/11.2.0
module load qt5/5.11.1
conda activate myeloma6 
module purge
module load slurm/current

export SWEEP_ID="nair-karthik00/MARS/iz3kbpyq"
export WANDB_API_KEY='e51b9830c285698aced24214131f58ee2bfbd5e3'
export WANDB_CACHE_DIR='/gpfs/home/kn2347/.cache/wandb'
export WANDB_CONFIG_DIR='/gpfs/home/kn2347/.config/wandb'
export WANDB_DIR='/gpfs/home/kn2347/wandb'

# Ensure SWEEP_ID is provided (from `wandb sweep` output)
if [[ -z "${SWEEP_ID:-}" ]]; then
  echo "ERROR: Please export SWEEP_ID before submitting. Run: export SWEEP_ID=<entity>/<project>/<sweep_id>"
  exit 1
fi

# Make a per-job log dir
LAUNCH_DIR="/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/other_prediction_heads/auxiliary_labels/tumor_volume_bin_50_sweep_logs/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LAUNCH_DIR}"

echo "Starting W&B agent for sweep ${SWEEP_ID} (array task ${SLURM_ARRAY_TASK_ID})"

# Some clusters set CUDA_VISIBLE_DEVICES automatically for single-GPU allocations; set it to 0 to be explicit
export CUDA_VISIBLE_DEVICES=0

# Run the agent and capture logs to file (foreground process)
wandb agent "${SWEEP_ID}" > "${LAUNCH_DIR}/wandb_agent.log" 2>&1

# agent will run until sweep finishes or is stopped
