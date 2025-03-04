#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --output=out/output_%A_%a.out
#SBATCH --error=err/error_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=netsi_standard
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --array=1-19  # Number of lines in params.txt

# Read the correct line from params.txt
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" lego_params.txt)

echo "Running job with parameters: ${PARAMS}"

# Run your script with the extracted parameters
python analyze_data.py $PARAMS
# python lattice_analyze_data.py

# eval python run_data_discrete.py $PARAMS
