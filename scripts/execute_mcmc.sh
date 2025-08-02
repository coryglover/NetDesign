#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --time=5-00:00:00
#SBATCH --job-name=Prot1
#SBATCH --partition=netsi_standard
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_%A_%a.log
#SBATCH --array=1-20%5

set -x

echo "hello from bash"

trap 'echo "Caught SIGTERM at $(date)"' TERM
trap 'echo "Caught SIGINT at $(date)"' INT
trap 'echo "Exited with code $?"' EXIT
 
# Read in parameters file
PARAMS=$(awk "NR==${SLURM_ARRAY_TASK_ID}" /projects/ccnr/glover.co/net_design/NetDesign/params/proteins/mcmc_params_1.txt)

sleep 1
echo "${PARAMS}"
sleep 1
echo "${SLURM_ARRAY_TASK_ID}"
# Run mcmc script with parameters
set -- $PARAMS
python /projects/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py "$@"
sleep 1
echo "job ${SLURM_ARRAY_TASK_ID} complete"
#python /work/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py --graph_file /scratch/glover.co/NetDesign/data/proteins/human/edgefiles/CPX-1919.edge --X_file /scratch/glover.co/NetDesign/data/proteins/human/Xfiles/X_CPX-1919.txt --num_samples 100000 --output /scratch/glover.co/NetDesign/data/proteins/human/treefiles 
