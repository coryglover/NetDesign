import os
import sys
import json
import numpy as np

# Get paths for molecules
molecule_path = '/scratch/glover.co/NetDesign/data/molecules/WHO'

lines_to_write = []

# molecule_edge_files = os.listdir(f'{molecule_path}/edgefiles')
with open('/scratch/glover.co/NetDesign/data/classifications/classA.json','r') as f:
    classA = json.load(f)
with open('/scratch/glover.co/NetDesign/data/classifications/classB.json','r') as f:
    classB = json.load(f)

molecule_edge_files = np.append(np.array(classA['molecules'])[:,2],np.array(classB['molecules'])[:,2])

for graph_file in molecule_edge_files:
    base_name = graph_file
    X_file = molecule_path + f'/Xfiles/X_{base_name}.txt'
    # add line to file
    line = f'--graph_file {molecule_path}/edgefiles/{graph_file}.edge --X_file {X_file} --num_samples 1000000 --output {molecule_path}/treefiles/ --c_exclusive'
    lines_to_write.append(line)

with open('mcmc_params.txt','w') as f:
    for line in lines_to_write:
        f.write(f"{line}\n")
