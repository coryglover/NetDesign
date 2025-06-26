#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --job-name=DF
#SBATCH --partition=short
#SBATCH --output=/scratch/glover.co/NetDesign/out/df_%A_%a.log
#SBATCH --error=/scratch/glover.co/NetDesign/err/df_%A_%a.log

python create_dataframe.py --dir /scratch/glover.co/NetDesign/data/proteins/
