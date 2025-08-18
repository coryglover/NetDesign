import numpy as np
import os

protein_path = '/scratch/glover.co/NetDesign/data/proteins'
molecule_path = '/scratch/glover.co/NetDesign/data/molecules'
robot_path = '/scratch/glover.co/NetDesign/data/robots'
circuit_path = '/scratch/glover.co/NetDesign/data/circuits'
ikea_path = '/scratch/glover.co/NetDesign/data/IkeaData'

paths = [protein_path, molecule_path, robot_path, circuit_path, ikea_path]
for p in paths:
    max_div = []
    subdirs = os.listdir(p)
    for subdir in subdirs:
        if subdir == 'stats':
            continue
        ikea_X_files = os.listdir(f'{p}/{subdir}/Xfiles')
        for graph_file in ikea_X_files:
            X_file = p + f'/{subdir}/Xfiles/{graph_file}'
            X = np.loadtxt(X_file)
            if len(X.shape) == 1:
                if len(X) == 1:
                    max_div.append(1)
                else:
                    max_div.append(0)
                continue
            elif X.shape[0] == X.shape[1]:
                max_div.append(1)
            else:
                max_div.append(0)
    np.savetxt(p+'/stats/max_diversity.txt',np.array(max_div))


