#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=infinite
#SBATCH --job-name=ProtDesign
#SBATCH --output=/scratch/glover.co/NetDesign/out/output_%A_%a.out
#SBATCH --error=/scratch/glover.co/NetDesign/err/error_%A_%a.err
#SBATCH --array=162

# Read the correct line from params.txt
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" protein_params_1.txt)

# echo "Running job with parameters: ${PARAMS}"

# Run experiment
python assembly_tree.py $PARAMS