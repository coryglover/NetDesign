import numpy as np
import networkx as nx
import sys
import importlib  # Standard Python module for reloading
sys.path.insert(0, '/work/ccnr/glover.co/net_design/NetDesign')
import network_design_cont_time as nd_cont
sys.path.insert(0, '/work/ccnr/glover.co/net_design/NetDesign')
import network_design as nd
import graph_tool as gt
from graph_tool import draw as gtdraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os

# Read in parameters
parser = argparse.ArgumentParser()
parser.add_argument('--file',type=str,help='Name of graph file')
parser.add_argument('--label_file',default='None',type=str,help='File with label matrix')
parser.add_argument('--ratio',type=float,help='Ratio between attachment and detachment rate')
parser.add_argument('--c',default = np.nan,type=float,help='Ratio between average attachment and detachment rate')
parser.add_argument('--starter_attachment',type=float,default=1,help='Underlying starter attachment rate')
parser.add_argument('--recon_file',type=str,help='File for reconstructed graph information')
parser.add_argument('--num_components',type=int,help='Number of copies to create')

args = parser.parse_args()
file = args.file
label_file = args.label_file
ratio = args.ratio
c = args.c
starter_attachment = args.starter_attachment
recon_file = args.recon_file
num_copies = args.num_components


# Ensure recon_file exists
try:
    os.mkdir(recon_file)
except:
    its_ok = True

# Read in graph
try:
    g = nx.read_edgelist(file)
except:
    g = nx.read_edgelist(file,delimiter=',',edgetype=str,nodetype=str)

# Extract 
O = nx.adjacency_matrix(g).toarray()

# Get label matrix
if label_file == 'None':
    X = np.eye(g.number_of_nodes())
else:
    X = np.loadtxt(label_file)

new_X = np.tile(X,reps=(num_copies,1))

test = nd.NetAssembly(new_X,O,new_X.sum(axis=1,dtype=int))

test.run(100000,link_strength=ratio)

nx.write_edgelist(test.g,f'{recon_file}/discrete_graph_{ratio}.txt')
# np.savetxt(f'{recon_file}/comp_size_{ratio}.txt',test.lccs)
# np.savetxt(f'{recon_file}/num_comp_{ratio}.txt',test.ccs)
# np.savetxt(f'{recon_file}/ts_{ratio}.txt',test.ts)