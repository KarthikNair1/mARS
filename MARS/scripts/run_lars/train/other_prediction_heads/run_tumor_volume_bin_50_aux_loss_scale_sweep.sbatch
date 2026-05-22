#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=tumor_vol50_sweep
#SBATCH --mem=100GB
#SBATCH --gres=gpu:4
#SBATCH --partition=a100_short
#SBATCH --time=3-00:00:00
#SBATCH --exclude=a100-4005,a100-4020
#SBATCH --output=/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/other_prediction_heads/auxiliary_labels/tumor_volume_bin_50_sweep-%j.out
#SBATCH --error=/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/other_prediction_heads/auxiliary_labels/tumor_volume_bin_50_sweep-%j.err

cd /gpfs/home/kn2347/MARS
set -x -euo pipefail

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

SWEEP_ID="${SWEEP_ID:-}"
if [[ -z "${SWEEP_ID}" ]]; then
    echo "ERROR: Please export SWEEP_ID before submitting this job."
    echo "Example: export SWEEP_ID=<entity>/<project>/<sweep_id>"
    exit 1
fi

NUM_AGENTS="${NUM_AGENTS:-${SLURM_GPUS_ON_NODE:-4}}"
if [[ "${NUM_AGENTS}" -gt "${SLURM_GPUS_ON_NODE:-4}" ]]; then
    echo "ERROR: NUM_AGENTS=${NUM_AGENTS} exceeds available GPUs (${SLURM_GPUS_ON_NODE:-4})."
    exit 1
fi

LAUNCH_DIR="/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/other_prediction_heads/auxiliary_labels/tumor_volume_bin_50_sweep_logs/${SLURM_JOB_ID}"
mkdir -p "${LAUNCH_DIR}"

echo "Launching sweep agents for ${SWEEP_ID}"
echo "Requested agents: ${NUM_AGENTS}"
echo "Logs in: ${LAUNCH_DIR}"

for gpu_id in $(seq 0 $((NUM_AGENTS - 1))); do
    agent_log="${LAUNCH_DIR}/agent_gpu${gpu_id}.log"
    echo "Starting agent on GPU ${gpu_id}, log: ${agent_log}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        wandb agent "${SWEEP_ID}"
    ) >"${agent_log}" 2>&1 &
done

wait
