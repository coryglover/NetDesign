import os
import sys

# Get paths for humans, mouse, and yeast
human_path = '/scratch/glover.co/NetDesign/data/proteins/human'
mouse_path = '/scratch/glover.co/NetDesign/data/proteins/mouse'
yeast_path = '/scratch/glover.co/NetDesign/data/proteins/yeast'
stats_path = '/scratch/glover.co/NetDesign/data/proteins/stats/'

lines_to_write = []

human_edge_files = os.listdir(f'{human_path}/edgefiles')
for graph_file in human_edge_files:
    base_name = graph_file[:-5]
    X_file = human_path + f'/Xfiles/X_{base_name}.txt'
    # add line to file
    line = f'--graph_file {human_path}/edgefiles/{graph_file} --X_file {X_file} --num_samples 1000000 --output {stats_path}'
    lines_to_write.append(line)

mouse_edge_files = os.listdir(f'{mouse_path}/edgefiles')
for graph_file in mouse_edge_files:
    base_name = graph_file[:-5]
    X_file = mouse_path + f'/Xfiles/X_{base_name}.txt'
    # add line to file
    line = f'--graph_file {mouse_path}/edgefiles/{graph_file} --X_file {X_file} --num_samples 1000 --output {stats_path}'
    lines_to_write.append(line)

yeast_edge_files = os.listdir(f'{yeast_path}/edgefiles')
for graph_file in yeast_edge_files:
    base_name = graph_file[:-5]
    X_file = yeast_path + f'/Xfiles/X_{base_name}.txt'
    # add line to file    
    line = f'--graph_file {yeast_path}/edgefiles/{graph_file} --X_file {X_file} --num_samples 1000 --output {stats_path}'
    lines_to_write.append(line)

with open('sp_params_1.txt','w') as f:
    for line in lines_to_write[:1000]:
        f.write(f"{line}\n")

with open('sp_params_2.txt','w') as f:
    for line in lines_to_write[1000:]:
        f.write(f"{line}\n")
