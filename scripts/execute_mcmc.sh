#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=5-00:00:00
#SBATCH --job-name=1919
#SBATCH --partition=netsi_standard
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_%A_%a.log


# Read in parameters file
#PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /projects/ccnr/glover.co/net_design/NetDesign/params/proteins/mcmc_params_2.txt)

#echo "${PARAMS}"
# Run mcmc script with parameters
#python /projects/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py $PARAMS
python /projects/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py --graph_file /scratch/glover.co/NetDesign/data/proteins/human/edgefiles/CPX-1919.edge --X_file /scratch/glover.co/NetDesign/data/proteins/human/Xfiles/X_CPX-1919.txt --num_samples 100000 --output /scratch/glover.co/NetDesign/data/proteins/human/treefiles 
