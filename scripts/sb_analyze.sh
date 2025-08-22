#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name=DF_Prot
#SBATCH --partition=short
#SBATCH --output=/scratch/glover.co/NetDesign/out/df_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/df_%A_%a.log
#SBATCH --array=1-619%1

set -x

echo "hello from bash"

echo "Job ran on node ${HOSTNAME}"

sleep 1

trap 'echo "Caught SIGTERM at $(date)" >> "$logfile"' TERM 
trap 'echo "Caught SIGINT at $(date)"' INT
trap 'echo "Exited with code $?"' EXIT
 
# Read in parameters file
PARAMS=$(awk "NR==${SLURM_ARRAY_TASK_ID}" /projects/ccnr/glover.co/net_design/NetDesign/params/proteins/analysis_params.txt)


sleep 1
echo "${PARAMS}"
sleep 1
echo "${SLURM_ARRAY_TASK_ID}"
# Run mcmc script with parameters
set -- $PARAMS
python /projects/ccnr/glover.co/net_design/NetDesign/scripts/analyze_data.py "$@"
#python /projects/ccnr/glover.co/net_design/NetDesign/scripts/max_diversity.py
sleep 1
echo "job ${SLURM_ARRAY_TASK_ID} complete"