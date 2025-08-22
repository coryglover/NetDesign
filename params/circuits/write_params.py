import os
import sys

# Get paths for humans, mouse, and yeast
circuit_path = '/scratch/glover.co/NetDesign/data/circuits'
lines_to_write = []

subdirs = os.listdir(circuit_path)
for subdir in subdirs:
    ikea_edge_files = os.listdir(f'{circuit_path}/{subdir}/edgefiles')
    for graph_file in ikea_edge_files:
        base_name = graph_file[:-5]
        X_file = circuit_path + f'/{subdir}/Xfiles/X_{base_name}.txt'
        # add line to file
        line = f'--graph_file {circuit_path}/{subdir}/edgefiles/{graph_file} --X_file {X_file} --num_samples 1000000 --output {circuit_path}/{subdir}/treefiles/'
        lines_to_write.append(line)

with open('mcmc_params_1.txt','w') as f:
    for line in lines_to_write[:1000]:
        f.write(f"{line}\n")

with open('mcmc_params_2.txt','w') as f:
    for line in lines_to_write[1000:2000]:
        f.write(f"{line}\n")

with open('mcmc_params_3.txt','w') as f:
    for line in lines_to_write[2000:]:
        f.write(f"{line}\n")
