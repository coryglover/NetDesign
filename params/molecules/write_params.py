import os
import sys
import json
import numpy as np

mcmc = False 

# Get paths for molecules
# molecule_path = '/scratch/glover.co/NetDesign/data/molecules/WHO'
molecule_path = '/Users/glover.co/Documents/laszlo/NetDesign/data/molecules'
lines_to_write = []

# Read networks to consider
with open('all_nets.txt', 'r') as f:
    networks_to_consider = [line.strip() for line in f.readlines()]


for j, net in enumerate(networks_to_consider):
    subdir = net.split('/')[0]
    graph_file = f'{molecule_path}/{net}'
    base_name = graph_file[:-5]
    X_file = molecule_path + f'/{subdir}/Xfiles/X_{base_name}.txt'
    # add line to file
    if mcmc:
        line = f'--graph_file {molecule_path}/{subdir}/edgefiles/{graph_file} --X_file {X_file} --num_samples 100000 --output {molecule_path}/{subdir}/treefiles/'
    else:
        line = f'--graph_file {molecule_path}/{subdir}/edgefiles/{graph_file} --X_file {X_file} --tree_file {molecule_path}/{subdir}/treefiles/{base_name}_tree.json --output {molecule_path}/stats/assembly_stats.csv'
    lines_to_write.append(line)

if mcmc:
    with open('mcmc_params.txt','w') as f:
        for line in lines_to_write:
            f.write(f"{line}\n")

else:
    with open('analysis_params.txt','w') as f:
        for line in lines_to_write:
            f.write(f"{line}\n")
