#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=mm_ablate
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short
#SBATCH --array=0-69%10
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

# If you use wandb, ensure credentials are available in your environment.
export WANDB_CACHE_DIR='/gpfs/home/kn2347/.cache/wandb'
export WANDB_CONFIG_DIR='/gpfs/home/kn2347/.config/wandb'
export WANDB_DIR='/gpfs/home/kn2347/wandb'

# ===== User-adjustable paths =====
BASE_CONFIG='/gpfs/home/kn2347/MARS/MARS/scripts/run_lars/train/train_mars_from_scratch/ablation_experiment/training_config.yaml'
SPLIT_DIR='/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/ablation_experiments'
#CHECKPOINT_DIR='/gpfs/data/mayerhoeferlab/projects/lars/lars-avg-max_chpt'
OUTPUT_ROOT='/gpfs/data/mayerhoeferlab/projects/myeloma/results/train/train_mars_from_scratch/ablation_experiment'
export BASE_CONFIG SPLIT_DIR OUTPUT_ROOT

# Keep in sync with your split-generation fractions.
FRACTIONS=(0.01 0.05 0.10 0.25 0.50 0.75 1.00)
EVAL_FRACTIONS=(0.10 0.10 0.10 0.25 0.50 0.75 1.00)
# T_0 cycle length scaled inversely with fraction so each model sees the same
# number of gradient steps per cosine cycle (base T_0=5 at fraction=1.0).
CYCLE_LENGTHS=(500 100 50 20 10 7 5)
RUN_VAL_EVERY=(5 5 5 4 2 1 1)

# Filenames that were generated in SPLIT_DIR for the fractions above.
SPLIT_FILES=(
  train_val_test_split_trainN_7.pkl
  train_val_test_split_trainN_37.pkl
  train_val_test_split_trainN_74.pkl
  train_val_test_split_trainN_185.pkl
  train_val_test_split_trainN_370.pkl
  train_val_test_split_trainN_554.pkl
  train_val_test_split_trainN_739.pkl
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEEDS_PER_FRACTION=10
FRACTION_INDEX=$((TASK_ID / SEEDS_PER_FRACTION))
SEED_INDEX=$((TASK_ID % SEEDS_PER_FRACTION))

FRACTION=${FRACTIONS[$FRACTION_INDEX]}
EVAL_FRACTION=${EVAL_FRACTIONS[$FRACTION_INDEX]}
SPLIT_FILE=${SPLIT_FILES[$FRACTION_INDEX]}
SPLIT_PATH="${SPLIT_DIR}/${SPLIT_FILE}"
#CHECKPOINT_PATH="${CHECKPOINT_DIR}/LARS_trained_state-dict_${SEED_INDEX}.pth"

CYCLE_LENGTH=${CYCLE_LENGTHS[$FRACTION_INDEX]}
export FRACTION_INDEX SEED_INDEX FRACTION SPLIT_FILE SPLIT_PATH CYCLE_LENGTH

if [[ ! -f "${SPLIT_PATH}" ]]; then
  echo "Missing split file: ${SPLIT_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/frac_${FRACTION}"

BEST_CKPT="${OUTPUT_ROOT}/frac_${FRACTION}/checkpoint_seed${SEED_INDEX}_run0_best.pth"
if [[ -f "${BEST_CKPT}" ]]; then
  echo "Skipping: best checkpoint already exists at ${BEST_CKPT}"
  exit 0
fi

# calculate runevaleverynepochs so that 
python /gpfs/home/kn2347/MARS/MARS/run_lars/train.py "${BASE_CONFIG}" \
  --seed_index "${SEED_INDEX}" \
  --output "${OUTPUT_ROOT}/frac_${FRACTION}" \
  --train_test_split_pkl "${SPLIT_PATH}" \
  --lr_scheduler_cycle_length "${CYCLE_LENGTH}" \
  --run_eval_every_n_epochs "${RUN_VAL_EVERY[$FRACTION_INDEX]}" \
  --eval_on_fraction_of_val "${EVAL_FRACTION}"
