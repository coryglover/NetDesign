#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=1:00:00
#SBATCH --job-name=write_ikea_params
#SBATCH --partition=netsi_standard
#SBATCH --output=/scratch/glover.co/NetDesign/out/write_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/write_%A_%a.log

# Run mcmc script with parameters
python write_specificity_params.py
