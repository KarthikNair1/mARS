#!/bin/bash
#SBATCH --job-name=nnunet_pipeline
#SBATCH --output=/gpfs/home/kn2347/MARS/logs/nnunet_pipeline_%j.out
#SBATCH --error=/gpfs/home/kn2347/MARS/logs/nnunet_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

set -euo pipefail

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
# Edit these for your dataset/model.
# If you already have a trained nnUNet model, leave RUN_TRAIN=0.
# If you only want inference on an existing model, set RUN_PREPROCESS=0 and RUN_TRAIN=0.

CONDA_ENV_NAME="autopet3_lesiontracer"

export nnUNet_raw="/gpfs/home/kn2347/nnUNet_raw"
export nnUNet_preprocessed="/gpfs/home/kn2347/nnUNet_preprocessed"
export nnUNet_results="/gpfs/home/kn2347/nnUNet_results"

DATASET_ID="999"
CONFIGS=("3d_fullres")
FOLDS=(0 1 2 3 4)

# nnUNet inference input folder.
# Must contain files named like CASEID_0000.nii.gz, CASEID_0001.nii.gz, ...
INPUT_NIFTI_DIR="/path/to/your/nnunet_input_nifti_folder"

# Output folder for predicted masks.
OUTPUT_ROOT="/path/to/your/nnunet_predictions"
export OUTPUT_ROOT

# Only needed if you train your own model in this job.
# Must follow the nnUNet raw dataset format with imagesTr/labelsTr/etc.
RAW_DATASET_DIR="/path/to/your/nnunet_raw_dataset"

# Optional postprocessing file if you have one from nnUNet results.
POSTPROCESSING_PKL=""
PLANS_JSON=""
DATASET_JSON=""

RUN_PREPROCESS=0
RUN_TRAIN=0
RUN_FIND_BEST=0
RUN_INFERENCE=1
RUN_POSTPROCESS=0
RUN_BURDEN=1

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
source ~/.bashrc
conda activate "${CONDA_ENV_NAME}"

mkdir -p "${OUTPUT_ROOT}" /gpfs/home/kn2347/MARS/logs

# ------------------------------------------------------------
# Optional: plan and preprocess
# ------------------------------------------------------------
if [[ "${RUN_PREPROCESS}" -eq 1 ]]; then
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity --no_pbar
fi

# ------------------------------------------------------------
# Optional: train model(s)
# ------------------------------------------------------------
if [[ "${RUN_TRAIN}" -eq 1 ]]; then
    for config in "${CONFIGS[@]}"; do
        for fold in "${FOLDS[@]}"; do
            nnUNetv2_train "${DATASET_ID}" "${config}" "${fold}" --npz
        done
    done
fi

# ------------------------------------------------------------
# Optional: identify best configuration
# ------------------------------------------------------------
if [[ "${RUN_FIND_BEST}" -eq 1 ]]; then
    nnUNetv2_find_best_configuration "${DATASET_ID}" -c "${CONFIGS[@]}"
fi

# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
if [[ "${RUN_INFERENCE}" -eq 1 ]]; then
    for config in "${CONFIGS[@]}"; do
        pred_dir="${OUTPUT_ROOT}/${config}"
        mkdir -p "${pred_dir}"
        nnUNetv2_predict \
            -i "${INPUT_NIFTI_DIR}" \
            -o "${pred_dir}" \
            -d "${DATASET_ID}" \
            -c "${config}"
    done
fi

# ------------------------------------------------------------
# Optional postprocessing
# ------------------------------------------------------------
if [[ "${RUN_POSTPROCESS}" -eq 1 ]]; then
    if [[ -z "${POSTPROCESSING_PKL}" ]]; then
        echo "POSTPROCESSING_PKL is empty; set it before enabling RUN_POSTPROCESS."
        exit 1
    fi
    for config in "${CONFIGS[@]}"; do
        pred_dir="${OUTPUT_ROOT}/${config}"
        post_dir="${OUTPUT_ROOT}/${config}_postprocessed"
        mkdir -p "${post_dir}"
        nnUNetv2_apply_postprocessing \
            -i "${pred_dir}" \
            -o "${post_dir}" \
            --pp_pkl_file "${POSTPROCESSING_PKL}" \
            -plans_json "${PLANS_JSON}" \
            -dataset_json "${DATASET_JSON}"
    done
fi

# ------------------------------------------------------------
# Tumor burden per scan
# ------------------------------------------------------------
if [[ "${RUN_BURDEN}" -eq 1 ]]; then
    python - <<'PY'
from pathlib import Path
import csv
import os
import numpy as np
import nibabel as nib

output_root = Path(os.environ["OUTPUT_ROOT"])
results_csv = output_root / "tumor_burden_per_scan.csv"

prediction_dirs = [p for p in output_root.iterdir() if p.is_dir()]
rows = []

for prediction_dir in prediction_dirs:
    for nii_path in sorted(prediction_dir.glob("*.nii.gz")):
        img = nib.load(str(nii_path))
        data = img.get_fdata(dtype=np.float32)
        voxel_volume_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
        tumor_volume_mm3 = float((data > 0).sum() * voxel_volume_mm3)
        rows.append({
            "prediction_dir": prediction_dir.name,
            "case_id": nii_path.name.replace(".nii.gz", ""),
            "tumor_volume_mm3": tumor_volume_mm3,
            "tumor_volume_ml": tumor_volume_mm3 / 1000.0,
            "voxel_volume_mm3": voxel_volume_mm3,
        })

with results_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["prediction_dir", "case_id", "tumor_volume_mm3", "tumor_volume_ml", "voxel_volume_mm3"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {results_csv}")
PY
fi
