import pandas as pd
import numpy as np
import networkx as nx
import sys
from matplotlib.colors import to_rgba
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/work/ccnr/glover.co/net_design/NetDesign')
import network_design as nd
import netrd
import matplotlib.pyplot as plt
import argparse
import os

# Accept input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--graph_dir',type=str,
                    default='../data/protein_complex/graphs/',
                    help='Directory for graphs')
parser.add_argument('--recon_dir',type=str,
                    default='../data/protein_complex/reconstructed_graphs/',
                    help='Directory for graphs')

args = parser.parse_args()
graph_dir = args.graph_dir
recon_dir = args.recon_dir

# Get list of graphs
graph_list = os.listdir(graph_dir)
for g_file in graph_list:
    g = nx.read_edgelist(f'{graph_dir}/{g_file}')
    if g.number_of_nodes() == 1:
        continue
    # Create labels
    X = np.eye(g.number_of_nodes())
    new_X = np.tile(X,reps=(5,1))
    O = nx.adjacency_matrix(g).toarray()

    # Set up environment
    obj = nd.NetAssembly(new_X,O,new_X.sum(axis=0,dtype=int))
    obj.run(10000,link_strength=.1)
    nx.write_edgelist(obj.g,f'{recon_dir}/01_{g_file}')

    # Set up environment
    obj = nd.NetAssembly(new_X,O,new_X.sum(axis=0,dtype=int))
    obj.run(10000,link_strength=1)
    nx.write_edgelist(obj.g,f'{recon_dir}/1_{g_file}')