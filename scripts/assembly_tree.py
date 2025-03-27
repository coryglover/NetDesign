"""
This file approximates an assembly tree by solving the minimum multicut problem on a graph.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from treelib import Node, Tree

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
    return g, cutset

def identify_sub_layers(node,level,node_order,X):
    """
    Identify sublayers of a node in an assembly tree.
    
    Parameters:
        node (int) - node in assembly tree
        
    Returns:
        list - sublayers of node
    """
    cut_edges = label_pairs(node,X,node_order[level])
    new_g, cutset = cut_graph(node, cut_edges)
    return list(nx.connected_components(new_g))

def measure_stability(X,O,ret_g=False,initial_graph=None,deg_cap=None):
    """
    Measure stability of a network.
    
    Parameters:
        X (ndarray) - matrix of node labels
        O (ndarray) - binding matrix
        
    Returns:
        int - stability index
    """
    cur_graphs = []
    if initial_graph is not None and initial_graph.number_of_nodes() == 1:
        if ret_g:
            return 1, initial_graph
        else:
            return 1
    for t in range(1000):
        test_g = microcanonical_ensemble(X,O,initial_graph=initial_graph,deg_cap=deg_cap)
        new = True
        for h in cur_graphs:
            if nx.is_isomorphic(h,test_g):
                new = False
                break
        if new:
            cur_graphs.append(test_g)
    if ret_g:
        return len(cur_graphs), cur_graphs[-1]
    else:
        return len(cur_graphs)

def create_disassembly_tree(g,X,O,deg_cap=None):
    """
    Create a disassembly tree for a network.
    
    Parameters:
        g (nx.Graph) - Graph
        X (ndarray) - matrix of node labels
        O (ndarray) - binding matrix

    Returns:
        Tree - disassembly tree
    """
    # Order nodes by importance
    node_importance = np.sum(O,axis=1)
    node_order = np.argsort(node_importance)[::-1]
    # Initialize assembly tree
    disassembly_tree = Tree()
    disassembly_tree.create_node(0,0)
    # Initialize dictionary for assembly nodes
    assembly_nodes = {0: list(g.nodes())}
    # Check if graph assembles to one network
    I = measure_stability(X,O,deg_cap=deg_cap)
    stability_dict = {0:I}
    if I == 1:
        return disassembly_tree, stability_dict, assembly_nodes
    # Break down tree until each root has stability 1
    leaves = disassembly_tree.leaves()
    leaf_stability = np.prod([stability_dict[i.identifier] for i in leaves])
    while leaf_stability > 1:
        # Find unstable leaves
        unstable_leaves = [i for i in leaves if stability_dict[i.identifier] > 1]
        # Iterate through unstable leaves
        for leaf in unstable_leaves:
            # Split leaves into sublayers
            sublayers = identify_sub_layers(nx.subgraph(g,assembly_nodes[leaf.identifier]).copy(),disassembly_tree.depth(node=leaf),node_order,X)
            # Add sublayers to tree
            for i, sublayer in enumerate(sublayers):
                disassembly_tree.create_node(disassembly_tree.size(),disassembly_tree.size(),parent=leaf.identifier)
                assembly_nodes[disassembly_tree.size() - 1] = sublayer
                # Update stability dictionary
                if len(sublayer) == 1:
                    stability_dict[disassembly_tree.size() - 1] = 1
                else:
                    stability_dict[disassembly_tree.size() - 1] = measure_stability(X[list(sublayer)],O,deg_cap=deg_cap)
        # Update leaves
        leaves = disassembly_tree.leaves()
        leaf_stability = np.prod([stability_dict[i.identifier] for i in leaves])
    return disassembly_tree, stability_dict, assembly_nodes

def approx_assembly_tree(g,X,O,deg_cap=None):
    """
    Approximate assembly tree by cutting highest connected patricles.
    
    Parameters:
        g (nx.Graph)
        X (ndarray) - matrix of node labels
        O (ndarray) - binding matrix
        
    Returns:
        nx.Graph - assembly tree
        I (int) - stability index
    """
    # Get disassembly tree
    disassembly_tree, stability_dict, assembly_nodes = create_disassembly_tree(g,X,O,deg_cap)
    # Initialize dictionary for assembly nodes
    assembly_graphs = {}
    # List nodes from leaves to root by depth
    nodes = disassembly_tree.all_nodes()
    depths = [disassembly_tree.depth(node=i) for i in nodes]
    nodes = [x for _, x in sorted(zip(depths,nodes))][::-1]
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
                print(g.nodes())
                leaf_g = microcanonical_ensemble(X,O,initial_graph=g,deg_cap=deg_cap)
            # Add graph to dictionary
            assembly_graphs[n.identifier] = leaf_g
            si[n.identifier] = 1
        else:
            # Combine graphs of children nodes
            child_graphs = [assembly_graphs[i.identifier] for i in child_nodes]
            # Combine graphs
            new_g = nx.compose_all(child_graphs)
            # Run simulation
            cur_si, new_g = measure_stability(X,O,initial_graph=new_g,ret_g=True,deg_cap=deg_cap)
            # Add graph to dictionary
            assembly_graphs[n.identifier] = new_g
            si[n.identifier] = cur_si
    return assembly_graphs, si, disassembly_tree



        


def create_capacity(X: np.ndarray, O: np.ndarray) -> dict:
    """
    Create a capacity dictionary based on node labels.

    Parameters:
        X (np.ndarray): One-hot encoded label matrix for nodes.
        O (np.ndarray): Adjacency-like matrix where O[i, j] indicates if a node of type i can connect to type j.

    Returns:
        dict: Capacities for each node.
    """
    capacities = {}
    node_labels = np.argmax(X, axis=1)  # Precompute labels for all nodes
    
    for i, label in enumerate(node_labels):
        # Use a dictionary comprehension to set capacities based on `O`
        capacities[i] = {k: int(O[label, k]) for k in range(O.shape[1])}
    
    return capacities

def microcanonical_ensemble(
    X,
    O,
    deg_cap=None,
    T = 10000,
    initial_graph = None,
    ret_H = False,
    directed = False,
    time_of_entry = None,
    multiedge = False,
    ret_cap = False
):
    """
    Generate a microcanonical ensemble for network design.

    Parameters:
        X (ndarray): Node assignment matrix (one-hot encoded).
        capacities (dict): Connection capacities for each node and label.
        ret_H (bool): Reserved parameter (currently unused).
        directed (bool): Whether the network is directed.
        time_of_entry (ndarray, optional): Entry times of nodes. Defaults to all zeros.
        self_loops (bool): Whether self-loops are allowed.
        ret_cap (bool): Whether to return updated capacities.

    Returns:
        nx.Graph or tuple: Generated graph, optionally with updated capacities.
    """
    # Initialize graph
    if initial_graph is not None:
        g = initial_graph
        nx.set_node_attributes(g, {i: int(np.argmax(X[i])) for i in range(len(X))}, 'label')
    else:
        g = nx.Graph()
        g.add_nodes_from(range(len(X)))
        # Add node attribute label
        nx.set_node_attributes(g, {i: int(np.argmax(X[i])) for i in range(len(X))}, 'label')
    
    if time_of_entry is None:
        time_of_entry = np.zeros(len(X), dtype=float)

    # # Generate and shuffle node pairs
    # if directed:
    #     node_pairs = np.array(list(permutations(g.nodes, 2)))
    # else:
    #     node_pairs = np.array(list(combinations(g.nodes, 2)))
    # np.random.shuffle(node_pairs)
    capacities = create_capacity(X,O)
    if initial_graph is not None:
        # Update capacities
        for u, v in g.edges():
            label0, label1 = g.nodes[u]['label'], g.nodes[v]['label']
            capacities[u][label1] -= 1
            capacities[v][label0] -= 1
    # Process node pairs
    labels = np.argmax(X, axis=1)  # Precompute node labels
    for t in range(T):
        # Randomly select node pair
        node0, node1 = np.random.choice(g.nodes(), size=2, replace=False)
        if not multiedge and node0 == node1:
            continue

        label0, label1 = labels[node0], labels[node1]
        edge0_capacity = capacities[node0].get(label1, 0)
        edge1_capacity = capacities[node1].get(label0, 0)

        if directed:
            if edge0_capacity > 0:
                if deg_cap is not None and g.degree(node0) == deg_cap[label0]:
                    continue
                g.add_edge(node0, node1)
                capacities[node0][label1] -= 1
        elif edge0_capacity > 0 and edge1_capacity > 0:
            if deg_cap is not None and (g.degree(node0) == deg_cap[label0] or g.degree(node1) == deg_cap[label1]):
                continue
            if not multiedge and g.has_edge(node0, node1):
                continue
            else:
                if g.has_edge(node0, node1):
                    # Increate edge weight
                    g[node0][node1]['weight'] += 1
                else:
                    g.add_edge(node0, node1, weight=1)
            capacities[node0][label1] -= 1
            capacities[node1][label0] -= 1

    return (g, capacities) if ret_cap else g

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

if __name__ == '__main__':
    # g = nx.Graph()
    # O = np.array([[0,1,1],[1,0,1],[1,1,1]])
    # X = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[1,0,0]])
    # A = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,1,1,0]])
    # g = nx.from_numpy_array(A)
    # nx.set_edge_attributes(g, 1, 'capacity')
    # draw_network(g,X,with_labels=True)
    # plt.show()

    # assembly_graphs, si = approx_assembly_tree(g,X,O)
    # print(si)
    # draw_network(assembly_graphs[0],X,with_labels=True)
    # plt.show()
    g = nx.read_edgelist('../data/protein_complex/proteins/yeast/edgefiles/CPX-1162.edge',nodetype=int)
    X = np.loadtxt('../data/protein_complex/proteins/yeast/Xfiles/X_CPX-1162.txt',dtype=int)
    nx.set_edge_attributes(g,1,'capacity')
    O = extract_O(g,X)
    draw_network(g,X,with_labels=True)
    plt.show()
    assembly_graphs, si, disassembly_tree = approx_assembly_tree(g,X,O)
    print(si)
    draw_network(assembly_graphs[0],X,with_labels=True)
    plt.show()
    print(nx.is_isomorphic(g,assembly_graphs[0]))
    disassembly_tree.show()
    # cut_edges = label_pairs(g,X,0)
    # new_g, cutset = cut_graph(g, cut_edges)
    # print(cutset)
    # draw_network(new_g,X,with_labels=True)
    # plt.show()
            

