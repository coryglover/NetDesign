#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --time=5-00:00:00
#SBATCH --job-name=Rob1
#SBATCH --partition=netsi_standard
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_%A_%a.log

params_file="/projects/ccnr/glover.co/net_design/NetDesign/params/robots/mcmc_params.txt"

count=3
max_count=10

while IFS= read -r line; do
	echo "Running with parameters: $line"
	python /projects/ccnr/glover.co/net_design/NetDesign/scripts/run_mcmc.py $line

	((count++))
        if [[ $count -ge $max_count ]]; then
            break
        fi
done < $params_file
