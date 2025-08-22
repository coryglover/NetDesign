import os
import sys

# Get paths for humans, mouse, and yeast
ikea_path = '/scratch/glover.co/NetDesign/data/IkeaData'
lines_to_write = []

subdirs = os.listdir(ikea_path)
for subdir in subdirs:
    if subdir == 'stats':
        continue
    ikea_edge_files = os.listdir(f'{ikea_path}/{subdir}/edgefiles')
    for graph_file in ikea_edge_files:
        base_name = graph_file[:-5]
        X_file = ikea_path + f'/{subdir}/Xfiles/X_{base_name}.txt'
        # add line to file
        line = f'--graph_file {ikea_path}/{subdir}/edgefiles/{graph_file} --X_file {X_file} --num_samples 1000000 --output /scratch/glover.co/NetDesign/data/IkeaData/stats/'
        lines_to_write.append(line)

with open('sp_params.txt','w') as f:
    for line in lines_to_write:
        f.write(f"{line}\n")
