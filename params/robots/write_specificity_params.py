import os
import sys

# Get paths for humans, mouse, and yeast
robot_path = '/scratch/glover.co/NetDesign/data/robots'
lines_to_write = []

subdirs = os.listdir(robot_path)
for subdir in subdirs:
    if subdir == 'stats':
        continue
    robot_edge_files = os.listdir(f'{robot_path}/{subdir}/edgefiles')
    for graph_file in robot_edge_files:
        base_name = graph_file[:-5]
        X_file = robot_path + f'/{subdir}/Xfiles/X_{base_name}.txt'
        O_file = robot_path + f'/{subdir}/Ofiles/O_{base_name}.txt'
        c_file = robot_path + f'/{subdir}/cfiles/c_{base_name}.txt'
        # add line to file
        line = f'--graph_file {robot_path}/{subdir}/edgefiles/{graph_file} --X_file {X_file} --num_samples 1000000 --output {robot_path}/stats/ --O_file {O_file} --c_file {c_file}'
        lines_to_write.append(line)

with open('sp_params.txt','w') as f:
    for line in lines_to_write:
        f.write(f"{line}\n")
