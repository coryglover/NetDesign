#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name=Sp_cir3
#SBATCH --partition=short
#SBATCH --output=/scratch/glover.co/NetDesign/out/sp_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/sp_%A_%a.log
#SBATCH --array=1-7%1

set -x

echo "hello from bash"

echo "Job ran on node ${HOSTNAME}"

sleep 1

trap 'echo "Caught SIGTERM at $(date)" >> "$logfile"' TERM 
trap 'echo "Caught SIGINT at $(date)"' INT
trap 'echo "Exited with code $?"' EXIT
 
# Read in parameters file
PARAMS=$(awk "NR==${SLURM_ARRAY_TASK_ID}" /projects/ccnr/glover.co/net_design/NetDesign/params/circuits/sp_params_3.txt)


sleep 1
echo "${PARAMS}"
sleep 1
echo "${SLURM_ARRAY_TASK_ID}"
# Run mcmc script with parameters
set -- $PARAMS
python /projects/ccnr/glover.co/net_design/NetDesign/scripts/specificity.py "$@"
#python /projects/ccnr/glover.co/net_design/NetDesign/scripts/max_diversity.py
sleep 1
echo "job ${SLURM_ARRAY_TASK_ID} complete"
#python /work/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py --graph_file /scratch/glover.co/NetDesign/data/proteins/human/edgefiles/CPX-1919.edge --X_file /scratch/glover.co/NetDesign/data/proteins/human/Xfiles/X_CPX-1919.txt --num_samples 100000 --output /scratch/glover.co/NetDesign/data/proteins/human/treefiles 
