#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --partition=netsi_standard
#SBATCH --time=2:00:00
#SBATCH --job-name=GenerateProteinGraphs
#SBATCH --output=out/proteingraphs.out
#SBATCH --error=err/proteingraphs.err

python make_protein_networks.py