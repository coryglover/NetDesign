#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --job-name=Prot2_simulation
#SBATCH --partition=short
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_%A_%a.log
#SBATCH --array=1-1000%10

# Read in parameters file
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /projects/ccnr/glover.co/net_design/NetDesign/params/proteins/mcmc_params_2.txt)

echo "${PARAMS}"
# Run mcmc script with parameters
python /projects/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py $PARAMS
