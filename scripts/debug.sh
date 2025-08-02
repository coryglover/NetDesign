#!/bin/bash
#SBATCH --job-name=array_srun
#SBATCH --array=1-5
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err

echo "Task ID: $SLURM_ARRAY_TASK_ID"
srun -p netsi_standard -w c3201 --pty bash
