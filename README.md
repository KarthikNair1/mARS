# MARS (Myeloma) Pipeline

This repository contains the MARS PET/CT pipeline used in this workspace:

1. DICOM preprocessing to H5 + metadata CSVs
2. 2D or 3D model training
3. Prediction generation on val/test splits
4. Aggregation + metric calculation

The commands below are based on the scripts/configs already present in this repo.

## Repository layout

- `MARS/preprocess_dicoms`: DICOM ingestion and preprocessing
- `MARS/run_lars`: 2D training/inference code
- `MARS/run_lars/code_3d`: 3D training/inference code
- `MARS/evaluate`: evaluation scripts
- `MARS/scripts`: runnable YAML / SLURM launcher scripts

## Environment assumptions

This project is currently path-coupled to GPFS paths in existing configs, e.g.:

- `/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars`
- `/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars`
- `/gpfs/data/mayerhoeferlab/projects/myeloma/results`

Typical environment setup from existing launchers:

```bash
module add anaconda3/gpu/2025.06
module load anaconda3/gpu/2025.06
module load gcc/11.2.0
module load qt5/5.11.1
conda activate myeloma6
```

For preprocessing, older launchers use `anaconda3/gpu/2022.10` with env `myeloma`.

If using Weights & Biases:

```bash
export WANDB_API_KEY="<your_wandb_key>"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_DIR="$HOME/wandb"
```

## 1) Preprocess DICOMs

### 1A. Convert DICOM studies to H5/PKL/CSV (array-style splitting)

Canonical script: `MARS/scripts/preprocess_dicoms/process_data.sbatch`

Equivalent direct command:

```bash
python -u /gpfs/home/kn2347/MARS/MARS/preprocess_dicoms/process_data.py \
  -N 50 \
  -n 0 \
  -n2 0 \
  -n3 -1 \
  --dataset_dir /gpfs/data/mayerhoeferlab/projects/myeloma/dicom \
  --save_dir /gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars
```

Notes:
- `-N` = number of splits/chunks
- `-n` = which split to process (used with `SLURM_ARRAY_TASK_ID` in sbatch)

### 1B. Build merged scan CSV and QC helpers

```bash
bash /gpfs/home/kn2347/MARS/MARS/scripts/preprocess_dicoms/combine_datafiles.sh
bash /gpfs/home/kn2347/MARS/MARS/scripts/preprocess_dicoms/check_axial_pixel_size.sh
bash /gpfs/home/kn2347/MARS/MARS/scripts/preprocess_dicoms/pick_out_scans.sh
```

These generate and curate datafiles under `process_dicoms_for_mars/datafiles`.

---

## 2) Train models

### 2A. Train 2D model

Base config example:
- `MARS/scripts/run_lars/train/train_mars_from_scratch/training_config.yaml`

Run:

```bash
python /gpfs/home/kn2347/MARS/MARS/run_lars/train.py \
  /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/train/train_mars_from_scratch/training_config.yaml
```

SLURM launcher example:
- `MARS/scripts/run_lars/train/train_mars_from_scratch/train.sbatch`

### 2B. Train 3D model

Base config example:
- `MARS/scripts/run_lars/train_3d/train_mars/training_config.yaml`

Run:

```bash
python /gpfs/home/kn2347/MARS/MARS/run_lars/code_3d/train.py \
  /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/train_3d/train_mars/training_config.yaml
```

SLURM launcher example:
- `MARS/scripts/run_lars/train_3d/train_mars/train.sbatch`

---

## 3) Run predictions

### 3A. 2D test prediction + aggregation + metrics

Canonical script:
- `MARS/scripts/run_lars/predictions/2D_all_seeds_prediction/eval_test.sh`

Equivalent command sequence:

```bash
cd /gpfs/home/kn2347/MARS

python /gpfs/home/kn2347/MARS/MARS/run_lars/predictions.py \
  /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/2D_all_seeds_prediction/config_test.yaml

python MARS/run_lars/aggregate_corsag.py \
  --pred_file_pattern "/gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_all_seeds_prediction/test/pred_test_*.csv"

python MARS/evaluate/calculate_metrics_given_predictions.py \
  --pred_csv /gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_all_seeds_prediction/test/pred_test_mipview_aggregate.csv \
  --results_save_dir /gpfs/data/mayerhoeferlab/projects/myeloma/results/2D_all_seeds_prediction/test/ \
  --train_test_split /gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl \
  --which_split test
```

### 3B. 3D test prediction + aggregation + metrics

Canonical script:
- `MARS/scripts/run_lars/predictions/3D_all_seeds_prediction/eval_test.sh`

Equivalent command sequence:

```bash
cd /gpfs/home/kn2347/MARS

python /gpfs/home/kn2347/MARS/MARS/run_lars/code_3d/predictions.py \
  /gpfs/home/kn2347/MARS/MARS/scripts/run_lars/predictions/3D_all_seeds_prediction/config_test.yaml

python MARS/run_lars/aggregate_corsag.py \
  --pred_file_pattern "/gpfs/data/mayerhoeferlab/projects/myeloma/results/3D_all_seeds_prediction/test/pred_test_*.csv"

python MARS/evaluate/calculate_metrics_given_predictions.py \
  --pred_csv /gpfs/data/mayerhoeferlab/projects/myeloma/results/3D_all_seeds_prediction/test/pred_test_mipview_aggregate.csv \
  --results_save_dir /gpfs/data/mayerhoeferlab/projects/myeloma/results/3D_all_seeds_prediction/test/ \
  --train_test_split /gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl \
  --which_split test
```

---

## 4) Key config locations

- 2D train configs: `MARS/scripts/run_lars/train/**/training_config*.yaml`
- 3D train configs: `MARS/scripts/run_lars/train_3d/**/training_config*.yaml`
- Prediction configs: `MARS/scripts/run_lars/predictions/**/config*.yaml`
- Preprocessing launchers: `MARS/scripts/preprocess_dicoms/*.sh` and `MARS/scripts/preprocess_dicoms/process_data.sbatch`

## Practical notes

- Many scripts assume absolute GPFS paths; copy a nearby YAML and edit paths for new runs.
- Output directories are created automatically by the Python entrypoints in most workflows.
- If you run from outside `/gpfs/home/kn2347/MARS`, use absolute paths exactly as in examples.
- `wandb/` under repo root mostly contains experiment artifacts; core code is under `MARS/`.
