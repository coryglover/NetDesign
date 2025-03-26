"""
The goal of this script is to exhaustively search all possible assembly trees for a given set of nodes.
It aims to find the tree which results in only one network and is on the shortest depth.
"""

import numpy as np
import networkx as nx
from itertools import combinations, chain

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
    # Process node pairs
    labels = np.argmax(X, axis=1)  # Precompute node labels
    for t in range(T):
        if len(g.nodes()) < 2:
            break
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

def calculate_current_capacities(g, X, O):
    """
    Calculate the current capacities of the network.

    Parameters:
        g (nx.Graph): Existing network graph.
        X (np.ndarray): One-hot encoded label matrix for nodes.
        O (np.ndarray): Adjacency-like matrix where O[i, j] indicates if a node of type i can connect to type j.

    Returns:
        dict: Updated capacities for each node.
    """
    capacities = create_capacity(X, O)
    labels = nx.get_node_attributes(g, 'label')
    
    for node0, node1 in g.edges():
        label0, label1 = labels[node0], labels[node1]
        capacities[node0][label1] -= 1
        capacities[node1][label0] -= 1
    
    return capacities

def continue_microcanonical_ensemble(
    g,
    X,
    O,
    deg_cap=None,
    T=10000,
    ret_H=False,
    directed=False,
    time_of_entry=None,
    multiedge=False,
    ret_cap=False
):
    """
    Continue running a microcanonical ensemble for network design on an existing network.

    Parameters:
        g (nx.Graph): Existing network graph.
        X (ndarray): Node assignment matrix (one-hot encoded).
        O (ndarray): Adjacency-like matrix where O[i, j] indicates if a node of type i can connect to type j.
        deg_cap (dict, optional): Degree capacities for each node label. Defaults to None.
        T (int, optional): Number of iterations. Defaults to 10000.
        ret_H (bool, optional): Reserved parameter (currently unused). Defaults to False.
        directed (bool, optional): Whether the network is directed. Defaults to False.
        time_of_entry (ndarray, optional): Entry times of nodes. Defaults to all zeros.
        multiedge (bool, optional): Whether multiple edges are allowed. Defaults to False.
        ret_cap (bool, optional): Whether to return updated capacities. Defaults to False.

    Returns:
        nx.Graph or tuple: Updated graph, optionally with updated capacities.
    """
    if time_of_entry is None:
        time_of_entry = np.zeros(len(X), dtype=float)

    capacities = calculate_current_capacities(g, X, O)
    labels = np.argmax(X, axis=1)  # Precompute node labels

    for t in range(T):
        if len(g.nodes()) < 2:
            break
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
                    # Increase edge weight
                    g[node0][node1]['weight'] += 1
                else:
                    g.add_edge(node0, node1, weight=1)
            capacities[node0][label1] -= 1
            capacities[node1][label0] -= 1

    return (g, capacities) if ret_cap else g

def estimate_num_nets(X,O,deg_cap=None,num_attempts=100,multiedge=True):
    """
    Estimate the number of networks that can be generated from a given set of nodes.

    Parameters:
        X (np.ndarray): One-hot encoded label matrix for nodes.
        O (np.ndarray): Adjacency-like matrix where O[i, j] indicates if a node of type i can connect to type j.

    Returns:
        int: Number of networks that can be generated.
    """
    count = 0
    graphs = []
    iso_count = {}
    for i in range(num_attempts):
        g = microcanonical_ensemble(X, O,deg_cap=deg_cap,multiedge=multiedge)
        found = False
        for j,h in enumerate(graphs):
            # Check whether graphs are isomorphic while maintaining node labels and edge weights
            if nx.is_isomorphic(g, h, node_match=lambda n1, n2: n1['label'] == n2['label'],edge_match=lambda e1, e2: e1['weight'] == e2['weight']):
                found = True
                iso_count[j] += 1
                break
        if not found:
            graphs.append(g)
            count += 1
            iso_count[len(iso_count.keys())] = 1
    return count, graphs, iso_count

def find_disjoint_sets(nodes):
    """
    Find all possible combinations of disjoint sets that make the full set.

    Parameters:
        nodes (list): List of nodes.

    Returns:
        list: List of tuples containing disjoint sets.
    """
    def all_subsets(lst):
        return chain(*map(lambda x: combinations(lst, x), range(1, len(lst))))

    nodes_set = set(nodes)
    disjoint_sets = []
    for subset in all_subsets(nodes):
        subset_set = set(subset)
        complement_set = nodes_set - subset_set
        if complement_set:
            disjoint_sets.append((list(subset_set), list(complement_set)))
    
    return disjoint_sets

def find_shallowest_tree(X, O, target_network, deg_cap=None):
    """
    Find the shallowest assembly tree which is guaranteed to assemble into the correct network.

    Parameters:
        X (np.ndarray): One-hot encoded label matrix for nodes.
        O (np.ndarray): Adjacency-like matrix where O[i, j] indicates if a node of type i can connect to type j.
        target_network (nx.Graph): The target network to be assembled.
        deg_cap (dict, optional): Degree capacities for each node label. Defaults to None.

    Returns:
        int: Depth of the shallowest assembly tree.
    """
    def recursive_assemble(subsets):
        if len(subsets) == 1:
            # Create a graph with just one node
            g = nx.Graph()
            g.add_node(subsets[0])
            nx.set_node_attributes(g, {subsets[0]: int(np.argmax(X[subsets[0]]))}, 'label')
            # Check if the final network is isomorphic to the target network
            if nx.is_isomorphic(g, target_network, node_match=lambda n1, n2: n1['label'] == n2['label'], edge_match=lambda e1, e2: e1['weight'] == e2['weight']):
                return 1
            else:
                return None
        
        for set1, set2 in find_disjoint_sets(subsets):
            X1 = np.vstack([X[i] for i in set1])
            X2 = np.vstack([X[i] for i in set2])
            O1 = O
            O2 = O
            num_networks1 = estimate_num_nets(X1, O1, deg_cap=deg_cap)
            num_networks2 = estimate_num_nets(X2, O2, deg_cap=deg_cap)
            if num_networks1 == 1 and num_networks2 == 1:
                combined_set = set1 + set2
                remaining_sets = [s for s in subsets if s not in combined_set]
                remaining_sets.append(combined_set)
                depth = recursive_assemble(remaining_sets)
                if depth is not None:
                    # Check if combining the two networks results in only one network
                    g1 = microcanonical_ensemble(X1, O1, deg_cap=deg_cap)
                    g2 = microcanonical_ensemble(X2, O2, deg_cap=deg_cap)
                    count = 0
                    for i in range(100):
                        g = continue_microcanonical_ensemble(g1, X2, O2, deg_cap=deg_cap)
                        if nx.is_isomorphic(g, g1, node_match=lambda n1, n2: n1['label'] == n2['label'], edge_match=lambda e1, e2: e1['weight'] == e2['weight']):
                            count += 1
                        else:
                            break
                    if count == 1:
                        # Check if the final network is isomorphic to the target network
                        final_network = continue_microcanonical_ensemble(g1, X2, O2, deg_cap=deg_cap)
                        if nx.is_isomorphic(final_network, target_network, node_match=lambda n1, n2: n1['label'] == n2['label'], edge_match=lambda e1, e2: e1['weight'] == e2['weight']):
                            return depth + 1
        return None

    nodes = list(range(len(X)))
    return recursive_assemble(nodes)

if __name__ == '__main__':
    X = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    O = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
    deg_cap = [1,2,3,4]
    target_network = nx.Graph()
    target_network.add_edges_from([(0, 5), (1, 5), (2, 4), (3, 4), (3, 5), (4,5)])
    print(find_shallowest_tree(X, O, target_network, deg_cap))