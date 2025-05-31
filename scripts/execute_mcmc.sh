#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --job-name=mcmc_simulation
#SBATCH --partition=short
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_output.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_error.log
#SBATCH --array=1%100

# Read in parameters file
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/glover.co/NetDesign/params/mcmc_params.txt)

echo $PARAMS
# Run mcmc script with parameters
python /projects/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py $PARAMS
