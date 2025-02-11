#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=20:00
#SBATCH --job-name=CleanData
#SBATCH --output=out/cleandata.out
#SBATCH --error=err/cleandata.err

python process_protein_data.py