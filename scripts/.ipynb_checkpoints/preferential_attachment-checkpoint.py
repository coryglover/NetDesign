import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# importing sys
import sys
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/work/ccnr/glover.co/net_design/NetDesign')
import network_design as nd
import time
from matplotlib.colors import to_rgba
import matplotlib.animation as animation
import os

def nonlinear_pa(N,m,alpha=1):
    """
    Generate a network with nonlinear preferential attachment

    Parameters:
        N (int) - number of nodes
        m (int) - number of links attached to incoming nodes
        alpha (float) - preferential attachment exponent

    Returns:
        nx.Graph
    """
    # Generate initial network
    g = nx.complete_graph(3)

    for i in range(3,N):
        # Get degree sequence
        deg_seq = np.array(g.degree())[:,1]
        # Get probabilities
        deg_alpha = np.power(deg_seq,alpha)
        prob = deg_alpha / np.sum(deg_alpha)
        
        # Add new node
        g.add_node(i)

        # Add link
        # Choose nodes
        links_to_add = np.random.choice(np.arange(i),p=prob,replace=False,size=m)
        # Add links
        for j in links_to_add:
            g.add_edge(i,j)

    A = nx.adjacency_matrix(g).toarray()
    node_order = np.arange(N)
    np.random.shuffle(node_order)
    A[np.arange(N),:] = A[node_order,:]
    A[:,np.arange(N)] = A[:,node_order]
    new_g = nx.Graph(A)
    for e in new_g.edges():
        del new_g[e[0]][e[1]]['weight']

    return new_g

# os.makedirs('../data/network_analysis/gnp/',exist_ok=True)

# gnp_comps = np.zeros((10,10))
# for k in range(1,11):
#     os.makedirs(f'../data/network_analysis/gnp/k_{k}/',exist_ok=True)
#     for i in range(10):
#         os.makedirs(f'../data/network_analysis/gnp/k_{k}/ex{i}/',exist_ok=True)
#         g = nx.fast_gnp_random_graph(100, k/100, directed=False)
#         # g = nx.k_core(g,1)
#         g = nx.subgraph(g,sorted(list(nx.connected_components(g)),key=len,reverse=True)[0])
#         X = np.eye(g.number_of_nodes())
#         O = nx.adjacency_matrix(g).toarray()
#         new_X = np.vstack((X,X))
#         random_network = nd.NetAssembly(new_X,O,new_X.sum(axis=0,dtype=int),system_energy=1)
#         random_network.run(10000,link_strength=.01)
#         nx.write_edgelist(g,f'../data/network_analysis/gnp/k_{k}/ex{i}/graph.txt')
#         nx.write_edgelist(random_network.g,f'../data/network_analysis/gnp/k_{k}/ex{i}/recon_graph.txt')
#         components = list(nx.connected_components(random_network.g))
#         for c in components:
#             if nx.is_isomorphic(g,nx.subgraph(random_network.g,c)):
#                 gnp_comps[i,k-1] += 1

# np.savetxt(f'../data/network_analysis/gnp/comps.txt',gnp_comps)

os.makedirs('../data/network_analysis/ba/',exist_ok=True)

ba_comps = np.zeros((10,21))
for j,alpha in enumerate(np.linspace(0,2,21)):
    if j < 14:
        continue
    os.makedirs(f'../data/network_analysis/ba/alpha_{alpha}/',exist_ok=True)
    for i in range(10):
        os.makedirs(f'../data/network_analysis/ba/alpha_{alpha}/ex_{i}/',exist_ok=True)
        g = nonlinear_pa(100,3,alpha)
        nx.write_edgelist(g,f'../data/network_analysis/ba/alpha_{alpha}/ex_{i}/g.txt')
        X = np.eye(g.number_of_nodes())
        new_X = np.tile(X,reps=(2,1))
        O = nx.adjacency_matrix(g).toarray()
        obj = nd.NetAssembly(new_X,O,new_X.sum(axis=0,dtype=int))
        obj.run(10000,link_strength=.01)
        nx.write_edgelist(obj.g,f'../data/network_analysis/ba/alpha_{alpha}/ex_{i}/recon_g.txt')
        components = list(nx.connected_components(obj.g))
        for c in components:
            if nx.is_isomorphic(g,nx.subgraph(obj.g,c)):
                ba_comps[i,j] += 1

np.savetxt(f'../data/network_analysis/ba/comps.txt',ba_comps)


    