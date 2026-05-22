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

cd /gpfs/home/kn2347/MARS

python /gpfs/home/kn2347/MARS/MARS/run_lars/predictions.py \
  /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/2D_all_seeds_prediction_using_LARS/config_val.yaml

python MARS/run_lars/aggregate_corsag.py \
  --pred_file_pattern '/gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_all_seeds_prediction_using_LARS/val/pred_val_*.csv'

python MARS/evaluate/calculate_metrics_given_predictions.py \
  --pred_csv /gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_all_seeds_prediction_using_LARS/val/pred_val_mipview_aggregate.csv \
  --results_save_dir /gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_all_seeds_prediction_using_LARS/val/ \
  --train_test_split /gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl \
  --which_split val
