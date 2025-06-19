"""
This file approximates an assembly tree by solving the minimum multicut problem on a graph.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from treelib import Node, Tree
import argparse
import os
import pickle
import copy
import json
from tqdm import tqdm

def cut_graph(g,pairs):
    cutset = set()
    # Cut graph to separate pairs
    for s, t in pairs:
        # Check whether path exists
        if nx.has_path(g, s, t):
            # Find cut of graph
            cut_value, partition = nx.minimum_cut(g, s, t)
            # Get cut set
            S, T = partition
            cut_edges = [(u, v) for u in S for v in T if g.has_edge(u, v)]
            cutset.update(cut_edges)
            g.remove_edges_from(cut_edges)
    return g

def identify_sub_layers(subset, node_to_sep, X, O, capacity, multiedge=False):
    """
    Identify sublayers of a node in an assembly tree.

    Parameters:
        subset (nx.Graph): Subgraph of the assembly tree.
        depth (int): Depth of the current node in the tree.
        node_order (list): Order of nodes by importance.
        X (ndarray): Matrix of node labels.
        O (ndarray): Binding matrix.
        capacity (ndarray): Capacity vector.
        multiedge (bool): Whether to allow multiple edges between nodes.

    Returns:
        list: Sublayers of the node.
    """
    # best_entropy = np.inf
    best_components = None

    cut_edges = label_pairs(list(subset.nodes()), X, node_to_sep)
    new_g = cut_graph(subset.copy(), cut_edges)

    if nx.is_connected(new_g):
        return None, 0

    components = list(nx.connected_components(new_g))
    valid_split = False

    for component in components:
        graphs = measure_stability(X[list(component)], O, capacity=capacity, multiedge=multiedge, ret_g=True)

        for h in graphs:
            if is_subgraph(subset,h):
                valid_split = True
                break

        # prob_dist_values, _, _ = prob_dist(X[list(component)], O, capacity, multiedge=multiedge)
        # cur_entropy += entropy(prob_dist_values)

    if valid_split:
        stability = len(graphs)
        best_components = components

    return best_components, stability

def canonical_form(G):
    # Fast graph hashing for isomorphism classes
    return nx.weisfeiler_lehman_graph_hash(G)

def count_isomorphs(graphs):
    counts = dict()
    for G in graphs:
        key = canonical_form(G)
        try:
            counts[key] += 1
        except:
            counts[key] = 1
    return counts

def O_respecting_rewiring(g,T):
    types = nx.get_node_attributes(g,"label")
    unique_types = np.unique(list(types.values()))
    k = unique_types.size

    class_to_vs = {type : [] for type in unique_types}
    for v in g:
        class_to_vs[types[v]].append(v)
    
    for i in range(T):
        t1,t2 = np.random.choice(unique_types,2)
        if t1 == t2 and len(class_to_vs[t1])>3:
            u1,u2,v1,v2 = np.random.choice(class_to_vs[t1],4,replace=False)
        elif t1 != t2 and len(class_to_vs[t1])>1 and len(class_to_vs[t2])>1:
            u1,u2 = np.random.choice(class_to_vs[t1],2,replace=False)
            v1,v2 = np.random.choice(class_to_vs[t2],2,replace=False)
        else:
            continue
        if g.has_edge(u1,v1) and g.has_edge(u2,v2) and not g.has_edge(u1,v2) and not g.has_edge(u2,v1):
            g.remove_edge(u1,v1)
            g.remove_edge(u2,v2)
            g.add_edge(u1,v2)
            g.add_edge(u2,v1)
    return g 

def rewire(g,X,O,T,fixed_edges=None):
    """
    Rewire a graph while respecting the binding matrix and node labels.
    Parameters:
        g (nx.Graph): Input graph.
        X (ndarray): Matrix of node labels.
        O (ndarray): Binding matrix.
        capacity (ndarray): Capacity vector.
        T (int): Number of rewiring iterations.

    Returns:
        nx.Graph: Rewired graph.
    """
    # Get adjacency matrix
    A = nx.adjacency_matrix(g).todense()
    nodes = list(g.nodes())
    pos_neighbors = X[nodes]@O - A@X[nodes]
    
    # Rewire the graph T times
    for i in range(T):
        # Choose random pair of nodes
        u = np.random.choice(g.nodes())
        # Choose other node of same type
        u_label = X[u].argmax()
        if np.sum(X[nodes,u_label]) == 1:
            continue
        v = np.random.choice([n for n in g.nodes() if X[n].argmax() == u_label and n != u])
        # Choose random neighbor of u
        u_neighbors = list(g.neighbors(u))
        if len(u_neighbors) == 0:
            continue
        u_neighbor = np.random.choice(u_neighbors)
        if (u,u_neighbor) in fixed_edges or (u_neighbor,u) in fixed_edges:
            continue
        # Get neighbor label
        u_neighbor_label = X[u_neighbor].argmax()
        # Update pos_neighbors
        pos_neighbors[nodes.index(u),u_neighbor_label] += 1
        # Get labels that could create connection
        possible_labels = np.where(pos_neighbors[nodes.index(u)] > 0)[0]
        # Get possible rewires for v
        v_neighbors = list(g.neighbors(v))
        # Get nodes that can be rewired to u
        possible_rewires = [n for n in v_neighbors if X[n].argmax() in possible_labels and n != u]
        # Randomize possible rewires
        possible_rewires = np.random.permutation(possible_rewires)
        # Try to rewire
        success = False
        for w in possible_rewires:
            # Update pos_neighbors
            w_label = X[w].argmax()
            if (v,w) in fixed_edges or (w,v) in fixed_edges:
                continue
            # Check whether rewired edges exist
            if (u,w) in g.edges() or (v,u_neighbor) in g.edges():
                continue
            pos_neighbors[nodes.index(w),u_label] += 1
            # Check if edge can be created
            if pos_neighbors[nodes.index(u),w_label] > 0 and pos_neighbors[nodes.index(w),u_label] > 0:
                # Check whether rewire will kill initial graph edge
                
                g.remove_edge(u,u_neighbor)
                g.remove_edge(v,w)
                g.add_edge(u,w)
                g.add_edge(v,u_neighbor)
                pos_neighbors[nodes.index(u),w_label] -= 1
                pos_neighbors[nodes.index(w),u_label] -= 1
                success = True
                break
            pos_neighbors[nodes.index(v),w_label] -= 1
        if not success:
            pos_neighbors[nodes.index(u),u_neighbor_label] -= 1
    # Return rewired graph
    return g

def prob_dist(X,O,capacity,max_iters=10,initial_graph=None,multiedge=False,verbose=False,labeled=False,T=1000,max_edges=False, rewire_est=True):
    """
    Extract empirical distribution of system.
    
    Parameters:
        X (ndarray) - matrix of node labels
        O (ndarray) - binding matrix
        
    Returns:
        int - stability index
    """
    cur_graphs = []
    if initial_graph is not None and initial_graph.number_of_nodes() == 1:
        return np.array([max_iters]), [initial_graph], np.array([0])
    if verbose:
        for t in tqdm(range(max_iters)):
            if (not rewire_est) or (t == 0):
                test_g, rates = microcanonical_ensemble(X,O,capacity,T=T,initial_graph=initial_graph.copy(),multiedge=multiedge,kappa_d=.1,ret_rates = True,max_edges=max_edges)
                if rates[:-test_g.number_of_nodes()].sum() != 0 and max_edges is False:
                    continue
                cur_graphs.append(test_g.copy())
            else:
                test_g = rewire(test_g.copy(),X,O,T=int(2*test_g.number_of_edges()),fixed_edges=list(initial_graph.edges()))
                cur_graphs.append(test_g.copy())
    else:
        for t in range(max_iters):
            if not rewire_est or t == 0:
                test_g, rates = microcanonical_ensemble(X,O,capacity,T=T,initial_graph=initial_graph.copy(),multiedge=multiedge,kappa_d=.1,ret_rates = True,max_edges=max_edges)
                if rates[:-test_g.number_of_nodes()].sum() != 0 and max_edges is False:
                    continue
                cur_graphs.append(test_g.copy())
            else:
                test_g = rewire(test_g.copy(),X,O,T=int(2*test_g.number_of_edges()),fixed_edges=list(initial_graph.edges()))
                cur_graphs.append(test_g.copy())
    final_graphs = []
    counts = []
    sorted_indices = np.array([])
    if verbose:
        for i in tqdm(range(len(cur_graphs))):
            g = cur_graphs[i]
            found = False
            if labeled:
                for idx in sorted_indices:
                    if np.allclose(nx.adjacency_matrix(g).todense(), nx.adjacency_matrix(final_graphs[idx]).todense()):
                        found = True
                        break
            else:
                for idx in sorted_indices:
                    if nx.is_isomorphic(g,final_graphs[idx]):
                        found = True
                        break
            if not found:
                final_graphs.append(g)
                counts.append(1)
            else:
                counts[idx] += 1
            # Reorder graphs based on number of counts
            counts = np.array(counts)
            sorted_indices = np.argsort(counts)[::-1]
            counts = list(counts)
    else:
        for i in range(len(cur_graphs)):
            g = cur_graphs[i]
            found = False
            if labeled:
                for idx in sorted_indices:
                    if np.allclose(nx.adjacency_matrix(g).todense(), nx.adjacency_matrix(final_graphs[idx]).todense()):
                        found = True
                        break
            else:
                for idx in sorted_indices:
                    if nx.is_isomorphic(g,final_graphs[idx]):
                        found = True
                        break
            if not found:
                final_graphs.append(g)
                counts.append(1)
            else:
                counts[idx] += 1
            # Reorder graphs based on number of counts
            counts = np.array(counts)
            sorted_indices = np.argsort(counts)[::-1]
            counts = list(counts)
    return np.array(counts), final_graphs, sorted_indices

def entropy(x):
    return - np.sum(x * np.log(x + 1e-10))

def is_subgraph(G, H):
    GM = nx.algorithms.isomorphism.GraphMatcher(G, H)
    return GM.subgraph_is_isomorphic()

def create_disassembly_tree(g, X, O, capacity=None, multiedge=False):
    """
    Create a disassembly tree for a network.

    Parameters:
        g (nx.Graph): Graph
        X (ndarray): Matrix of node labels
        O (ndarray): Binding matrix

    Returns:
        Tree: Disassembly tree
    """
    # Order nodes by importance
    node_importance = np.sum(O, axis=1)
    node_order = np.argsort(node_importance)[::-1]
    # Initialize assembly tree
    disassembly_tree = Tree()
    disassembly_tree.create_node(0, 0)
    # Initialize dictionary for assembly nodes
    assembly_nodes = {0: list(g.nodes())}
    # Check if graph assembles to one network
    graphs = measure_stability(X, O, capacity=capacity,multiedge=multiedge, ret_g=True)
    I = len(graphs)
    stability_dict = {0: I}
    if I == 1 and nx.is_isomorphic(g, graphs[0]):
        return disassembly_tree, stability_dict, assembly_nodes
    # Break down tree until each root has stability 1
    leaves = disassembly_tree.leaves()
    leaf_stability = np.prod([stability_dict[i.identifier] for i in leaves])
    k = 0
    # Find unstable leaves
    unstable_leaves = [i for i in leaves if stability_dict[i.identifier] > 1]
    # Iterate through unstable leaves
    for leaf in unstable_leaves:
        same = True
        sublayers, stability = identify_sub_layers(
                nx.subgraph(g, assembly_nodes[leaf.identifier]).copy(),
                node_order[k],
                X,
                O,
                capacity
            )
        k += 1
        k = k % len(node_order)
        if sublayers is None:
            continue
        # Add sublayers to tree
        for i, sublayer in enumerate(sublayers):
            disassembly_tree.create_node(
                disassembly_tree.size(),
                disassembly_tree.size(),
                parent=leaf.identifier,
            )
            assembly_nodes[disassembly_tree.size() - 1] = sublayer
            # Update stability dictionary
            stability_dict[disassembly_tree.size() - 1] = stability
        # Update leaves
        leaves = disassembly_tree.leaves()
        for l in leaves:
            if l not in unstable_leaves:
                unstable_leaves.append(l)
    # leaf_stability = np.prod([stability_dict[i.identifier] for i in leaves])
    return disassembly_tree, stability_dict, assembly_nodes

def approx_assembly_tree(g, X, O, capacity=None, multiedge=False):
    """
    Approximate assembly tree by cutting highest connected particles.

    Parameters:
        g (nx.Graph)
        X (ndarray): Matrix of node labels
        O (ndarray): Binding matrix

    Returns:
        nx.Graph: Assembly tree
        I (int): Stability index
    """
    # Get disassembly tree
    disassembly_tree, stability_dict, assembly_nodes = create_disassembly_tree(
        g, X, O, capacity, multiedge
    )
    # Initialize dictionary for assembly nodes
    assembly_graphs = {}
    # List nodes from leaves to root by depth
    nodes = disassembly_tree.all_nodes()
    depths = [disassembly_tree.depth(node=i) for i in nodes]
    nodes = [x for _, x in sorted(zip(depths, nodes))][::-1]
    si = {}
    # Iterate through nodes
    for n in nodes:
        # Get children of current node
        child_nodes = disassembly_tree.children(n.identifier)
        if child_nodes == []:
            # Get assembly nodes
            cur_nodes = assembly_nodes[n.identifier]
            # Create empty graph with nodes
            g = nx.Graph()
            g.add_nodes_from(cur_nodes)
            if len(cur_nodes) == 1:
                leaf_g = g
            else:
                # Generate network
                leaf_g = microcanonical_ensemble(X, O, initial_graph=g, capacity=capacity, multiedge=multiedge)
            # Add graph to dictionary
            assembly_graphs[n.identifier] = leaf_g
            si[n.identifier] = 1
        else:
            # Combine graphs of children nodes
            child_graphs = [assembly_graphs[i.identifier] for i in child_nodes]
            # Combine graphs
            new_g = nx.compose_all(child_graphs)
            # Run simulation
            cur_si, new_g = measure_stability(
                X, O, initial_graph=new_g, ret_g=True, capacity=capacity, multiedge=multiedge
            )
            # Add graph to dictionary
            assembly_graphs[n.identifier] = new_g
            si[n.identifier] = cur_si
    return assembly_graphs, si, disassembly_tree

def microcanonical_ensemble(
    X,
    O,
    capacity,
    kappa_a=1.0,
    kappa_d=0.01,
    T = 10000,
    max_iters = int(10e6),
    initial_graph = None,
    multiedge = False,
    ret_rates = False,
    max_edges = False
):
    """
    Generate a microcanonical ensemble draw using Gillepsie algorithm.

    Parameters:
        X (ndarray): Matrix of node labels.
        O (ndarray): Binding matrix.
        capacity (ndarray): Capacity vector.
        kappa_a (float): Attachment rate.
        kappa_d (float): Detachment rate.
        T (int): Number of iterations.
        max_iters (int): Maximum number of iterations.
        initial_graph (networkx.Graph): Initial graph to start from.
        multiedge (bool): Whether to allow multiple edges between nodes.
    """
    # Initialize graph
    if initial_graph is None:
        N = X.shape[0]
        if multiedge:
            g = nx.MultiGraph()
            g.add_nodes_from(np.arange(N))
        else:
            g = nx.Graph()
            g.add_nodes_from(np.arange(N))
    else:
        g = initial_graph
    # Get true edges which should always exist
    true_edges = list(g.copy().edges())
    # Check that number of labels and binding matrix is the same
    if X.shape[1] != O.shape[0]:
        raise ValueError("Number of labels and binding matrix do not match.")
    nodes = list(g.nodes())
    X = X[nodes]
    labels = X.argmax(axis=1)
    # Get adjacency matrix
    A = nx.adjacency_matrix(g).todense()
    # Initialize variables
    if max_edges:
        cur_edges = 0
        cur_graph = g.copy()
    
    N = g.number_of_nodes()
    t = 0
    counter = 0
    potential_links = (X@O - A@X)@X.T
    # Account for exist links in initial graph
    compatibility = np.heaviside(potential_links,0.0).astype(int)
    
    # Initialize rates
    rates_attach = compatibility[np.triu_indices(N)] * compatibility.T[np.triu_indices(N)] * kappa_a
    rates_detach = kappa_d * np.array([1 - g.degree(j) / (capacity[labels[i]] + 1) for i, j in enumerate(g.nodes())])
    rates = np.concatenate((rates_attach, rates_detach))
    # Begin simulation
    while t < T and counter < max_iters:
        counter += 1
        # Draw two uniform random variables
        u1, u2 = np.random.uniform(0, 1, 2)
        # Make sure simulation doesn't run too long
        if np.sum(rates_detach) == 0 or counter > max_iters:
            break
        # Calculate time step
        dt = -np.log(u1) / np.sum(rates)
        t += dt

        # Draw event
        event = np.searchsorted(np.cumsum(rates) / np.sum(rates), u2)
        # Attachment event
        if event < len(rates_attach):
            # Get nodes involved
            idx_i = np.triu_indices(N)[0][event]
            idx_j = np.triu_indices(N)[1][event]
            i = nodes[idx_i]
            j = nodes[idx_j]
            # Get node labels
            i_label = labels[idx_i]
            j_label = labels[idx_j]
            # Check that i and j are in the graph
            if i not in g.nodes() or j not in g.nodes():
                continue
            if i == j:
                continue
            if g.degree(i) == capacity[i_label] or g.degree(j) == capacity[j_label]:
                continue
            # Add edge
            if multiedge:
                g.add_edge(i,j)
            else:
                if g.has_edge(i,j):
                    continue
                g.add_edge(i,j)
        
            

            # Update potential links
            potential_links[idx_i,labels==j_label] -= 1
            potential_links[idx_j,labels==i_label] -= 1
            # Check whether nodes are at capacity
            if g.degree(i) == capacity[i_label]:
                potential_links[idx_i,:] = 0
            if g.degree(j) == capacity[j_label]:
                potential_links[idx_j,:] = 0 

            # Update compatibility
            compatibility = np.heaviside(potential_links,0.0).astype(int)
            # Update rates
            rates_attach = compatibility[np.triu_indices(N)] * compatibility.T[np.triu_indices(N)] * kappa_a

            rates_detach[idx_i] = kappa_d * (1 - g.degree(i) / (capacity[i_label]+1))
            rates_detach[idx_j] = kappa_d * (1 - g.degree(j) / (capacity[j_label]+1))
            # Update rates
            rates = np.concatenate((rates_attach, rates_detach))
            # print('Attach',i,j,rates_detach[i],rates_detach[j])
            if max_edges:
                if g.number_of_edges() >= cur_edges:
                    cur_edges = g.number_of_edges()
                    cur_graph = g.copy()
                    # print('Max edges',cur_edges)
        # Detachment event
        else:
            # Get node
            idx_i = event - len(rates_attach)
            i = nodes[idx_i]
            if i not in g.nodes():
                continue
            i_label = labels[idx_i]
            # Make node isolate
            neighbors = list(g.neighbors(i))
            for j in neighbors:
                g.remove_edge(i,j)
                # Get index of node j
                idx_j = nodes.index(j)
                # Update potential links
                # potential_links[idx_i,labels==labels[idx_j]] += 1
                # potential_links[idx_j,labels==i_label] += 1
                # Update compatibility
                compatibility = np.heaviside(potential_links,0.0).astype(int)
                rates_detach[idx_j] = kappa_d * (1 - g.degree(j) / (capacity[labels[idx_j]]+1))

            missing_edges = [edge for edge in true_edges if not g.has_edge(*edge)]
            g.add_edges_from(missing_edges)
            # Get current potential links
            potential_links = (X@O - nx.adjacency_matrix(g).todense()@X)@X.T
            # Update compatibility
            compatibility = np.heaviside(potential_links,0.0).astype(int)
            # Update rates
            rates_attach = compatibility[np.triu_indices(N)] * compatibility.T[np.triu_indices(N)] * kappa_a
            rates_detach[idx_i] = kappa_d * (1 - g.degree(i) / (capacity[i_label]+1))
            rates = np.concatenate((rates_attach, rates_detach))
            # If links are removed make sure true edges still exist
            

            # print('Detach',i,rates_detach[i])
    if ret_rates:
        if max_edges:
            return cur_graph, rates
        return g, rates
    else:
        if max_edges:
            return cur_graph
        return g

def draw_network(g,X,colors=None,**kwargs):
    """
    Draw network where nodes are colored based on their labels.
    
    g (networkx) - Graph
    X (ndarray) - matrix of node labels
    """
    if colors is None:
        # Create colormap ranging across rainbow based on label
        colormap = plt.cm.rainbow(np.linspace(0, 1, X.shape[1]))
        colors = [colormap[X[i].argmax()] for i in g.nodes()]
    # Make width of edges proportional to number of multiedges
    try:
        edge_width = [d['weight'] for (u,v,d) in g.edges(data=True)]
    except:
        edge_width = 1
    nx.draw(g, node_color=colors, width=edge_width, **kwargs)
    pass

def label_pairs(nodes,X,label_id):
    """
    Return pairs of nodes with a given label.
    
    g (networkx) - Graph
    X (ndarray) - matrix of node labels
    label_id (int) - label to search for
    """
    return list(combinations([i for i in nodes if np.argmax(X[i])==label_id],2))

def extract_O(g,X):
    """
    Extract binding matrix from network G and label matrix X.

    Parameters:
        g (networkx) - Graph
        X (ndarray) - label matrix

    Returns:
        O (ndarray) - binding matrix
    """
    # Get number of particle types
    particle_num = X.shape[1]
    labels = X.argmax(axis=1)

    # Initialize binding matrix
    O = np.zeros((particle_num,particle_num))

    # Iterate through each node and its neighbors
    for node in sorted(g.nodes):
        # Get node label
        node_label = labels[node]
        # Get neighbor labels
        neighbor_labels = labels[list(g.neighbors(node))]
        # Count neighbor labels
        counts = np.bincount(neighbor_labels, minlength=particle_num)
        # Update O matrix
        O[node_label] = np.maximum(O[node_label], counts)

    return O

def extract_deg_cap(g,X):
    """
    Extract binding matrix from network G and label matrix X.

    Parameters:
        g (networkx) - Graph
        X (ndarray) - label matrix

    Returns:
        O (ndarray) - binding matrix
    """
    # Get number of particle types
    particle_num = X.shape[1]
    labels = X.argmax(axis=1)

    # Initialize binding matrix
    deg_cap = np.zeros((particle_num,1))

    # Iterate through each node and its neighbors
    for node in sorted(g.nodes):
        # Get node label
        node_label = labels[node]
        # Get degree
        deg = g.degree[node]
        # Update O matrix
        deg_cap[node_label,0] = np.maximum(deg_cap[node_label,0], deg)

    return deg_cap

def measure_stability(X, O, ret_g=False, initial_graph=None, capacity=None,multiedge=False):
    """
    Measure stability of a network.

    Parameters:
        X (ndarray): Matrix of node labels.
        O (ndarray): Binding matrix.
        ret_g (bool): Whether to return the last generated graph.
        initial_graph (nx.Graph, optional): Initial graph to start from.
        deg_cap (dict, optional): Degree capacities for each node label.

    Returns:
        int: Stability index.
        nx.Graph (optional): Last generated graph if ret_g is True.
    """
    cur_graphs = []
    if initial_graph is not None and initial_graph.number_of_nodes() == 1:
        if ret_g:
            return 1, initial_graph
        else:
            return 1
    for t in range(1000):
        test_g = microcanonical_ensemble(X, O, initial_graph=initial_graph, capacity=capacity, multiedge=multiedge, kappa_d=1, T=1)
        new = True
        for h in cur_graphs:
            if nx.is_isomorphic(h, test_g):
                new = False
                break
        if new:
            cur_graphs.append(test_g)
    return cur_graphs
    
if __name__ == '__main__':
    X = np.vstack([np.eye(3) for i in range(2)])
    O = np.array([[0,1,1],[1,0,1],[1,1,2]])
    capacity = O.sum(axis=1,dtype=int)
    target = nx.Graph()
    target.add_nodes_from(np.arange(6))
    target.add_edges_from([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[2,5]])
    new_g = microcanonical_ensemble(X,O,capacity)
    draw_network(new_g,X,with_labels=True)
    for i in range(10):
        new_g = rewire(new_g,X,O,T=1000)
        draw_network(new_g,X,with_labels=True)