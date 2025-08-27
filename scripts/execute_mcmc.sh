#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=7-00:00:00
#SBATCH --job-name=Prot
#SBATCH --partition=netsi_largemem
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_%A_%a.log
#SBATCH --array=1-619%20
##SBATCH --exclude=c3105,c3016,c3107,c3108,c3109,c3110,c3111,c3112,c3113,c3114,c3115,c3116
set -x

echo "hello from bash"

echo "Job ran on node ${HOSTNAME}"

sleep 1

trap 'echo "Caught SIGTERM at $(date)" >> "$logfile"' TERM 
trap 'echo "Caught SIGINT at $(date)"' INT
trap 'echo "Exited with code $?"' EXIT
 
# Read in parameters file
PARAMS=$(awk "NR==${SLURM_ARRAY_TASK_ID}" /projects/ccnr/glover.co/net_design/NetDesign/params/proteins/mcmc_params.txt)


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
