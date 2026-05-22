#!/bin/bash
#SBATCH --job-name=nnunet_predict
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --partition=a100_short
#SBATCH --array=0-48
#SBATCH --output=nnunet_predict_%a_%j.log

# Source conda
module add anaconda3/gpu/2025.06
module load anaconda3/gpu/2025.06
module load gcc/11.2.0
module load qt5/5.11.1
conda activate autopet3_lesiontracer
module purge
module load slurm/current

INPUT_DIR="/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/nibabel_rewrite_for_autopet3_flat"
OUTPUT_DIR="/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars/autopet3_predictions"
MODEL_DIR="/gpfs/scratch/kn2347/MARS/nnUNet_results/Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"

export nnUNet_raw="/gpfs/data/mayerhoeferlab/projects/myeloma/nnUNet_raw"
export nnUNet_preprocessed="/gpfs/scratch/kn2347/MARS/nnUNet_preprocessed"
export nnUNet_results="/gpfs/scratch/kn2347/MARS/nnUNet_results"

# Each task runs on a separate node with 1 GPU, always use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Create per-task input/output subdirectories with unique names
GPU_INPUT_DIR="/tmp/nnunet_input_task_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
GPU_OUTPUT_DIR="/tmp/nnunet_output_task_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
rm -rf "$GPU_INPUT_DIR" "$GPU_OUTPUT_DIR"
mkdir -p "$GPU_INPUT_DIR" "$GPU_OUTPUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID (on separate node)"
echo "Input: $GPU_INPUT_DIR"
echo "Output: $GPU_OUTPUT_DIR"

# Get list of unique cases (keys) by looking for _0000 files only
mapfile -t CASES < <(find "$INPUT_DIR" -maxdepth 1 -name '*_0000.nii.gz' -type f | sed 's/_0000.nii.gz$//' | xargs -I {} basename {} | sort)
TOTAL_CASES=${#CASES[@]}
TOTAL_TASKS=$SLURM_ARRAY_TASK_COUNT

echo "Total cases: $TOTAL_CASES, Task ID: $SLURM_ARRAY_TASK_ID, Total tasks: $TOTAL_TASKS"

# Divide cases (pairs) into chunks: each task gets a disjoint set of complete pairs
CASES_PER_TASK=$(( (TOTAL_CASES + TOTAL_TASKS - 1) / TOTAL_TASKS ))
START_IDX=$(( SLURM_ARRAY_TASK_ID * CASES_PER_TASK ))
END_IDX=$(( START_IDX + CASES_PER_TASK ))
if (( END_IDX > TOTAL_CASES )); then
    END_IDX=$TOTAL_CASES
fi

echo "Processing cases $START_IDX to $((END_IDX-1)) ($(( END_IDX - START_IDX )) cases, $(( (END_IDX - START_IDX) * 2 )) files)"

# Symlink this task's files (complete pairs only)
for ((i=START_IDX; i<END_IDX; i++)); do
    CASE="${CASES[$i]}"
    ln -s "$INPUT_DIR/${CASE}_0000.nii.gz" "$GPU_INPUT_DIR/${CASE}_0000.nii.gz"
    ln -s "$INPUT_DIR/${CASE}_0001.nii.gz" "$GPU_INPUT_DIR/${CASE}_0001.nii.gz"
done

# Run nnUNet prediction
echo "Starting nnUNet prediction for task $SLURM_ARRAY_TASK_ID on GPU 0..."
nnUNetv2_predict_from_modelfolder \
    -i "$GPU_INPUT_DIR" \
    -o "$GPU_OUTPUT_DIR" \
    -m "$MODEL_DIR" \
    -device cuda \
    --verbose

EXIT_CODE=$?

# Copy results back to shared output directory
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Task $SLURM_ARRAY_TASK_ID: Prediction succeeded, copying results..."
    mkdir -p "$OUTPUT_DIR"
    find "$GPU_OUTPUT_DIR" -maxdepth 1 -name '*.nii.gz' -exec mv {} "$OUTPUT_DIR/" \;
    echo "Task $SLURM_ARRAY_TASK_ID: Results copied to $OUTPUT_DIR"
else
    echo "Task $SLURM_ARRAY_TASK_ID: Prediction failed with exit code $EXIT_CODE"
    # Keep GPU_OUTPUT_DIR for debugging if needed
    exit $EXIT_CODE
fi

# Cleanup temp directories only on success
rm -rf "$GPU_INPUT_DIR"
if [[ $EXIT_CODE -eq 0 ]]; then
    rm -rf "$GPU_OUTPUT_DIR"
else
    echo "Keeping $GPU_OUTPUT_DIR for debugging (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
