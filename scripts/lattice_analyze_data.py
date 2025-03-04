import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgba
import sys
import os
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/work/ccnr/glover.co/net_design/NetDesign')
import network_design as nd
sys.path.insert(0, '/work/ccnr/glover.co/net_design/NetDesign')
import network_design_cont_time as nd_cont
import netrd
import copy
import json
import argparse

distances = {
    # 'Jaccard':                 netrd.distance.JaccardDistance(),
    # 'Hamming':                 netrd.distance.Hamming(),
    # # 'HammingIpsenMikhailov':   netrd.distance.HammingIpsenMikhailov(),
    # 'Frobenius':               netrd.distance.Frobenius(),
    'PolynomialDissimilarity': netrd.distance.PolynomialDissimilarity()}
    # 'DegreeDivergence':        netrd.distance.DegreeDivergence(),
    # 'PortraitDivergence':      netrd.distance.PortraitDivergence(),
    # 'QuantumJSD':              netrd.distance.QuantumJSD(),
    # 'CommunicabilityJSD':      netrd.distance.CommunicabilityJSD(),
    # 'GraphDiffusion':          netrd.distance.GraphDiffusion(),
    # 'ResistancePerturbation':  netrd.distance.ResistancePerturbation(),
    # 'NetLSD':                  netrd.distance.NetLSD(),
    # 'IpsenMikhailov':          netrd.distance.IpsenMikhailov(),
    # # 'NonBacktrackingSpectral': netrd.distance.NonBacktrackingSpectral(),
    # # 'DistributionalNBD':       netrd.distance.DistributionalNBD(),
    # # 'DMeasure':                netrd.distance.DMeasure(),
    # 'DeltaCon':                netrd.distance.DeltaCon(),
    # 'NetSimile':               netrd.distance.NetSimile()}

def component_level_diff(g,assembled_g,f=nx.is_isomorphic,ignore_isolates=True):
    """
    Measures some function on the connected components of assembled network and compares with G.

    Parameters:
        g (nx.Graph) - original graph
        assembled_g (nx.Graph) - new graph

    Returns:
        metric (np.array) - metric between connected components in assembled_g and g
    """
    # Get connected components
    components = sorted(list(nx.connected_components(assembled_g)),key=len,reverse=True)
    metrics = np.zeros(len(components))

    # Loop through components
    for i, c in enumerate(components):
        # Check whetehr component is an isolate
        if len(c) == 1 and ignore_isolates:
            metrics[i] = np.nan
        metrics[i] = f(g,nx.subgraph(assembled_g,c))

    return metrics[~np.isnan(metrics)]

def perfect_assembly(g,assembled_g,C=2):
    """
    Checks whether the assembled graph is a perfect assembly of C copies of g

    Parameters:
        g (nx.Graph) - original graph
        assembled_g (nx.Graph) - assembled graph
        C (int) - number of copies of g

    Returns:
        bool - True if perfect assembly
    """
    return nx.is_isomorphic(make_copies(g,C),assembled_g)

def make_copies(g,C):
    """
    Make C copies of g

    Parameters:
        g (nx.graph) 
        C (int) - number of copies

    Returns:
        g_copy
    """
    copy_g = copy.copy(g)
    for i in range(2,C+1):
        copy_g = nx.disjoint_union(copy_g,g)
    return copy_g

def calc_stats(g,X):
    N = g.number_of_nodes()
    L = g.number_of_edges()
    comp = X.shape[1]
    return N, L, 2*L/N, comp / N, (L - N + 1)/L

# Read in parameters
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,help='directory holding data')
parser.add_argument('--save_dir',type=str,help='File to save analysis to')

args = parser.parse_args()
data_dir = '/work/ccnr/glover.co/net_design/NetDesign/data/lattice_data/cubic_graph_recon_2/'
save_dir = '/work/ccnr/glover.co/net_design/NetDesign/data/analysis/lattice_data/cubic_graph_recon_2/'
g = nx.read_edgelist('/work/ccnr/glover.co/net_design/NetDesign/data/lattice_data/cubic_graph.txt',nodetype=int)
# Get list of files in diectory
files = os.listdir(data_dir)
N = len(files)

detach_vals = np.array(files,dtype=float)
num_iso_comp = np.zeros((N,10))
perf_iso = np.zeros((N,10))
num_comp = np.zeros((N,10))
distance_metrics = np.zeros((N,10,len(distances)))
distance_metrics_size_controlled = np.zeros((N,10,len(distances)))
for i in range(N):
    for j in range(10):
        # Read in network
        assembled_g = nx.read_edgelist(data_dir+f'{files[i]}/' + f'ex_{j}/g.txt')

        # Reorder assembled_g by size of connected component
        components = sorted(list(nx.connected_components(assembled_g)),key=len,reverse=True)
        comp_sizes = np.array([len(x) for x in components]) / assembled_g.number_of_nodes()
        # max_mask = sum([list(i) for i in sorted(nx.connected_components(g),key=len)],[])[::-1]
        # assembled_g = nx.relabel_nodes(assembled_g, {j:i for i,j in enumerate(max_mask)})

        # Calculate number of isomorphic components
        num_iso_comp[i,j] = np.nan_to_num(component_level_diff(g,assembled_g,f=nx.is_isomorphic)).sum()
        perf_iso[i,j] = perfect_assembly(g,assembled_g)
        num_comp[i,j] = len(components)
        for k, d in enumerate(distances.values()):
            try:
                vals = component_level_diff(g,assembled_g,f=d)
                distance_metrics[i,j,k] = vals.mean()
                distance_metrics_size_controlled[i,j,k] = (vals * comp_sizes).sum()
            except:
                distance_metrics[i,j,k] = np.nan
                distance_metrics_size_controlled[i,j,k] = np.nan

# Save files
try:
    os.mkdir(save_dir)
except:
    already_made = True

np.savetxt(f'{save_dir}/num_iso_comp.txt',num_iso_comp)
np.savetxt(f'{save_dir}/num_comp.txt',num_comp)
np.savetxt(f'{save_dir}/perf_iso.txt',perf_iso)
np.save(f'{save_dir}/distance_metrics.npy',distance_metrics)
np.save(f'{save_dir}/distance_metrics_size_controlled.npy',distance_metrics_size_controlled)