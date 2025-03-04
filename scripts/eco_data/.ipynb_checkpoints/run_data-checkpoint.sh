#!/bin/bash
#SBATCH --job-name=small_eco_array
#SBATCH --output=/work/ccnr/glover.co/net_design/NetDesign/scripts/out/output_%A_%a.out
#SBATCH --error=/work/ccnr/glover.co/net_design/NetDesign/scripts/err/error_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --array=1-22%100  # Number of lines in params.txt

# Read the correct line from params.txt
# for i in $(seq 0 7); do
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" high_cyc_ex_params.txt)

echo "Running job with parameters: ${PARAMS}"

# Run your script with the extracted parameters
python run_data_cont.py $PARAMS
# done
# eval python run_data_discrete.py $PARAMS
