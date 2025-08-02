#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --time=5-00:00:00
#SBATCH --job-name=Ikea1
#SBATCH --partition=netsi_standard
#SBATCH --output=/scratch/glover.co/NetDesign/out/mcmc_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/mcmc_%A_%a.log


