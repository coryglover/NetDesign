import os
import sys

mcmc = False 

# Get paths for humans, mouse, and yeast
# robot_path = '/Users/glover.co/Documents/laszlo/NetDesign/data/robots'
robot_path = '/scratch/glover.co/NetDesign/data/robots'
lines_to_write = []

# Read networks to consider
with open('all_nets.txt', 'r') as f:
    networks_to_consider = [line.strip() for line in f.readlines()]


for j, net in enumerate(networks_to_consider):
    subdir = net.split('/')[0]
    graph_file = f'{robot_path}/{net}'
    base_name = graph_file.split('/')[-1][:-5]
    X_file = robot_path + f'/{subdir}/Xfiles/X_{base_name}.txt'
    # add line to file
    if mcmc:
        line = f'--graph_file {graph_file} --X_file {X_file} --num_samples 100000 --output {robot_path}/{subdir}/treefiles/ --O_file {robot_path}/{subdir}/Ofiles/O_{base_name}.txt --c_file {robot_path}/{subdir}/cfiles/c_{base_name}.txt'
    else:
        line = f'--graph_file {graph_file} --X_file {X_file} --tree_file {robot_path}/{subdir}/treefiles/{base_name}_tree.json --output {robot_path}/stats/assembly_stats.csv --O_file {robot_path}/{subdir}/Ofiles/O_{base_name}.txt --c_file {robot_path}/{subdir}/cfiles/c_{base_name}.txt'
    lines_to_write.append(line)

if mcmc:
    with open('mcmc_params.txt','w') as f:
        for line in lines_to_write:
            f.write(f"{line}\n")

else:
    with open('analysis_params.txt','w') as f:
        for line in lines_to_write:
            f.write(f"{line}\n")

