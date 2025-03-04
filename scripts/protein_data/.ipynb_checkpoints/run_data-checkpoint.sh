#!/bin/bash
#SBATCH --job-name=protein_array
#SBATCH --output=out/output_%A_%a.out
#SBATCH --error=err/error_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=netsi_standard
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --array=1-22%100  # Number of lines in params.txt

# Read the correct line from params.txt
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" low_cyc_low_div_params_1.txt)

echo "Running job with parameters: ${PARAMS}"

# Run your script with the extracted parameters
python run_data_cont.py $PARAMS

# eval python run_data_discrete.py $PARAMS
