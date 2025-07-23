#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10GB
#SBATCH --time=5-00:00:00
#SBATCH --job-name=C1Q
#SBATCH --partition=netsi_standard
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_%A_%a.log
#SBATCH --array=1-20%5

# Read in parameters file
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /projects/ccnr/glover.co/net_design/NetDesign/params/proteins/mcmc_params_1.txt)

sleep 1
echo "${PARAMS}"
which python
# Run mcmc script with parameters
python /projects/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py $PARAMS
sleep 1
echo "job ${SLURM_ARRAY_TASK_ID} complete"
#python /work/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py --graph_file /scratch/glover.co/NetDesign/data/proteins/human/edgefiles/CPX-1919.edge --X_file /scratch/glover.co/NetDesign/data/proteins/human/Xfiles/X_CPX-1919.txt --num_samples 100000 --output /scratch/glover.co/NetDesign/data/proteins/human/treefiles 
