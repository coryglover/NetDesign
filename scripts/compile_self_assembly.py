import numpy as np
import os

protein_path = '/scratch/glover.co/NetDesign/data/proteins/stats/'
molecule_path = '/scratch/glover.co/NetDesign/data/molecules/WHO/stats/'
robot_path = '/scratch/glover.co/NetDesign/data/robots/stats/'
circuit_path = '/scratch/glover.co/NetDesign/data/circuits/stats/'
ikea_path = '/scratch/glover.co/NetDesign/data/IkeaData/stats/'

paths = [protein_path, molecule_path, robot_path, circuit_path, ikea_path]
for p in paths:
    files = os.listdir(p)
    self_assembly = np.zeros(len(files) - 1)
    counter = 0
    for i in range(len(files)):
        if files[i] == 'specificity.txt' or files[i] == 'sa.txt':
            continue
        print(np.loadtxt(p+files[i]))
        self_assembly[counter] = int(np.loadtxt(p + files[i]))
        counter += 1
    np.savetxt(f'{p}sa.txt',self_assembly)
