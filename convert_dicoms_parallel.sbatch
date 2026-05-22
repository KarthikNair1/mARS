#!/bin/bash
#SBATCH --job-name=dicom_to_nifti
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --array=0-1600%50
#SBATCH --output=dicom_conversion_%a_%j.log

# Source conda
module add anaconda3/gpu/2025.06
module load anaconda3/gpu/2025.06
module load gcc/11.2.0
module load qt5/5.11.1
conda activate autopet3_lesiontracer
module purge
module load slurm/current

INPUT_DIR="/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/dicoms_rewrite"
OUTPUT_DIR="/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/nibabel_rewrite_for_autopet3_flat"
SCRIPT="/gpfs/home/kn2347/MARS/convert_dicom_to_nifti.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get the subject directory for this array task
mapfile -t SUBJECTS < <(ls -d "$INPUT_DIR"/*/ | sort)
TOTAL_SUBJECTS=${#SUBJECTS[@]}

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_SUBJECTS ]]; then
    echo "Task $SLURM_ARRAY_TASK_ID exceeds total subjects ($TOTAL_SUBJECTS), exiting"
    exit 0
fi

SUBJECT_DIR="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
SUBJECT_NAME=$(basename "$SUBJECT_DIR")

echo "Task $SLURM_ARRAY_TASK_ID processing: $SUBJECT_NAME"

python "$SCRIPT" "$SUBJECT_DIR" "$OUTPUT_DIR"
EXIT_CODE=$?

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $EXIT_CODE"
exit $EXIT_CODE
