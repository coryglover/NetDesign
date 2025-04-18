#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=4:00:00
#SBATCH --job-name=C1Q
#SBATCH --output=/scratch/glover.co/NetDesign/out/c1q.out
#SBATCH --error=/scratch/glover.co/err/NetDesign/c1q.err

python execute_assembly.py