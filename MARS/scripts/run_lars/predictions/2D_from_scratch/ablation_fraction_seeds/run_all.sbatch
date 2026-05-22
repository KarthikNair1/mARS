#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=mm_ablate_eval
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short
#SBATCH --array=0-6
#SBATCH --time=03:00:00
#SBATCH --exclude=a100-4005,a100-4020

set -euo pipefail
set -x

cd /gpfs/home/kn2347/MARS

module add anaconda3/gpu/2025.06
module load anaconda3/gpu/2025.06
module load gcc/11.2.0
module load qt5/5.11.1
# Conda activation scripts can reference unset tool vars; disable nounset temporarily.
set +u
conda activate myeloma6
set -u
module purge
module load slurm/current

export WANDB_CACHE_DIR='/gpfs/home/kn2347/.cache/wandb'
export WANDB_CONFIG_DIR='/gpfs/home/kn2347/.config/wandb'
export WANDB_DIR='/gpfs/home/kn2347/wandb'

# ===== User-adjustable paths =====
BASE_CONFIG='/gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/2D_from_scratch/ablation_fraction_seeds/config_test.yaml'
TRAIN_OUTPUT_ROOT='/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/train_mars_from_scratch/ablation_experiment'
PRED_OUTPUT_ROOT='/gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_from_scratch/ablation_fraction_seeds/test'
SPLIT_PKL='/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl'

# Keep in sync with the training fractions.
FRACTIONS=(0.01 0.05 0.10 0.25 0.50 0.75 1.00)

FRACTION=${FRACTIONS[${SLURM_ARRAY_TASK_ID}]}
MODEL_PATTERN="${TRAIN_OUTPUT_ROOT}/frac_${FRACTION}/checkpoint_seed*_run0_best.pth"
PRED_OUTPUT="${PRED_OUTPUT_ROOT}/frac_${FRACTION}"

mkdir -p "${PRED_OUTPUT}"

# --- 1. Generate predictions for all seeds of this fraction ---
python MARS/run_lars/predictions.py "${BASE_CONFIG}" \
  --model_path_pattern "${MODEL_PATTERN}" \
  --output "${PRED_OUTPUT}"

# --- 2. Aggregate across seeds ---
python MARS/run_lars/aggregate_corsag.py \
  --pred_file_pattern "${PRED_OUTPUT}/pred_test_*.csv"

# --- 3. Compute test metrics ---
python MARS/evaluate/calculate_metrics_given_predictions.py \
  --pred_csv "${PRED_OUTPUT}/pred_test_mipview_aggregate.csv" \
  --results_save_dir "${PRED_OUTPUT}" \
  --train_test_split "${SPLIT_PKL}" \
  --which_split test
