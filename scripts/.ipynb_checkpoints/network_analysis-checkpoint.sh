#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=4-00:00:00
#SBATCH --job-name=NetworkAnalysis
#SBATCH --output=out/na.out
#SBATCH --error=err/na.err

python preferential_attachment.py