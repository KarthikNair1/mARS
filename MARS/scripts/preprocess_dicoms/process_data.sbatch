#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=process_dicoms
#SBATCH --mem=16GB
#SBATCH --array=0-49
#SBATCH --partition=cpu_medium
#SBATCH --time=1-00:00:00

cd /gpfs/home/kn2347/MARS
set -x -e

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
echo SLURM_NTASKS=$SLURM_NTASKS

module purge
module add anaconda3/gpu/2022.10
module load anaconda3/gpu/2022.10
module load gcc/11.2.0
conda activate myeloma
module purge
module load slurm/current

export WANDB_API_KEY='e51b9830c285698aced24214131f58ee2bfbd5e3'
export WANDB_CACHE_DIR='/gpfs/home/kn2347/.cache/wandb'
export WANDB_CONFIG_DIR='/gpfs/home/kn2347/.config/wandb'
export WANDB_DIR='/gpfs/home/kn2347/wandb'

export CUDA_VISIBLE_DEVICES='0'

export dataset_dir='/gpfs/data/mayerhoeferlab/projects/myeloma/dicom'
export save_dir='/gpfs/data/mayerhoeferlab/projects/myeloma/process_dicoms_for_mars'
export N=50
export n=$SLURM_ARRAY_TASK_ID
export n2=0
export n3=-1

python -u /gpfs/home/kn2347/MARS/MARS/preprocess_dicoms/process_data.py \
    -N ${N} \
    -n ${n} \
    -n2 ${n2} \
    -n3 ${n3} \
    --dataset_dir ${dataset_dir} \
    --save_dir ${save_dir} \
> ${save_dir}/log_for_${SLURM_JOB_ID}_${n}.log 2>&1