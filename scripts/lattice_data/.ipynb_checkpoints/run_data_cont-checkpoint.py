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
parser.add_argument('--detachment_rate',type=float,help='Detachment rate')
parser.add_argument('--c',default = np.nan,type=float,help='Ratio between average attachment and detachment rate')
parser.add_argument('--starter_attachment',type=float,default=1,help='Underlying starter attachment rate')
parser.add_argument('--recon_file',type=str,help='File for reconstructed graph information')
parser.add_argument('--num_components',type=int,help='Number of copies to create')

args = parser.parse_args()
file = args.file
label_file = args.label_file
detachment_rate = args.detachment_rate
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
g = nx.read_edgelist(file,nodetype=int)

# Extract 
# O = nx.adjacency_matrix(g).toarray()

# Get label matrix
# Make file for given detachment rate
try:
    os.mkdir(os.path.join(recon_file,str(1/detachment_rate)))
except:
    its_ok = True

for k in range(10):
    try:
        os.mkdir(os.path.join(os.path.join(recon_file,str(1/detachment_rate)),f'ex_{k}'))
    except:
        its_ok = True
    if label_file == 'None':
        X = np.eye(g.number_of_nodes())
    else:
        X = np.loadtxt(label_file)
    X = np.ones(g.number_of_nodes()).reshape(g.number_of_nodes(),1)
    
    new_X = np.tile(X,reps=(num_copies,1))
    O = nd.extract_O(g,X)
    test = nd_cont.network(new_X,O)
    
    if c == 0:
        test.run(100000,link_attractiveness=starter_attachment,link_resilience=detachment_rate,relative_rates=None)
    else:
        test.run(100000,link_attractiveness=None,link_resilience=None,relative_rates=c)
    
    nx.write_edgelist(test.g,f'{recon_file}/{str(1/detachment_rate)}/ex_{k}/g.txt')
    np.savetxt(f'{recon_file}/{str(1/(detachment_rate))}/ex_{k}/comp_size.txt',test.lccs)
    np.savetxt(f'{recon_file}/{str(1/(detachment_rate))}/ex_{k}/num_comp.txt',test.ccs)
    np.savetxt(f'{recon_file}/{str(1/(detachment_rate))}/ex_{k}/ts.txt',test.ts)