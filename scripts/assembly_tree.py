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
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
import random
import time
import mcmc

def load_tree(f,tree,parent=None):
    nodes_to_add = list(f.keys())
    if 'success' in nodes_to_add:
        nodes_to_add.remove('success')        
    while len(nodes_to_add) > 0:
        node = nodes_to_add.pop(0)
        if parent is not None:
            tree.Tree.create_node(int(node), int(node), parent=parent, data=mcmc.AssemblyNode(f[node]['data'],tree.X,tree.O,tree.capacity))
        # Check whether node has children
        if 'children' in f[str(node)]:
            children = f[str(node)]['children']
            for child in children:
                load_tree(child, tree, parent=int(node))
        tree.update_prob(int(node))
    return tree

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

def find_optimal_edge_count(X, O, capacity, old_sol=None, initial_graph=None, solution=True, disp=False, ret_edges=False):
    """
    Using linear programming, find the optimal number of edges in a microcanonical ensemble graph
    given the node features X, the edge features O, and the capacity constraints.

    Parameters
    ----------
    X (ndarray): Label matrix
    O (ndarray): Binding matrix
    capacity (ndarray): Capacity vector
    old_sol (ndarray, optional): Previous solution to exclude.
    initial_graph (networkx.Graph, optional): Initial graph to start the optimization from.
    solution (bool): If True, return the solution vector, otherwise return the number of edges.
    disp (bool): If True, display the optimization process.
    ret_edges (bool): If True, return the edges of the optimal graph.

    Returns
    -------
    int or ndarray: The optimal number of edges in the microcanonical ensemble graph.
    """
    # Get number of nodes and possible edges
    if initial_graph is None:
        initial_graph = nx.Graph()
        initial_graph.add_nodes_from(N)
        N = X.shape[0]    
    nodes = list(initial_graph.nodes())
    N = len(nodes)
    pos_edges = N*(N-1) // 2
    

    # Create the constraint matrix
    capacity_constraints = np.zeros((N,pos_edges))
    O_constraints = np.zeros((O.shape[0]*N,pos_edges))
    idx_i, idx_j = np.triu_indices(N,k=1,m=N)
    for i in range(N):
        for j in range(len(idx_i)):
            if idx_i[j] == i or idx_j[j] == i:
                capacity_constraints[i,j] = 1
                for k in range(O.shape[0]):
                    if idx_i[j] == i:
                        O_constraints[i*O.shape[0]+k,j] = X[nodes[idx_j[j]],k]
                    else:
                        O_constraints[i*O.shape[0]+k,j] = X[nodes[idx_i[j]],k]
    constraint_mat = np.vstack([capacity_constraints, O_constraints])
    # Remove constraints that refer to existing edges
    edges = np.array(list(initial_graph.edges()))
    edges_idx = []
    for e1, e2 in edges:
        # Get index of nodes in node list
        idx_e1 = nodes.index(e1)
        idx_e2 = nodes.index(e2)
        # Find (e1,e2) in idx_i and idx_j
        edges_idx.append(int(np.where((idx_i == idx_e1) & (idx_j == idx_e2))[0][0]))
    # Remove columns associated with existing edges
    constraint_mat = np.delete(constraint_mat, edges_idx, axis=1)

    # Add constraint to ensure the new solution differs from old_sol
    if old_sol is not None:
        b_u = np.hstack((X[nodes,:]@capacity - np.array([initial_graph.degree[i] for i in nodes]),(X[nodes,:]@O - nx.adjacency_matrix(initial_graph).todense()@X[nodes,:]).flatten()))
  # Ensure at least one difference
        for sol in old_sol:
            diff_constraint = np.ones((1, pos_edges))
            diff_constraint[0, :] = sol
            constraint_mat = np.vstack([constraint_mat, diff_constraint])
            # edge_constraint = np.ones((1, pos_edges))
            # edge_constraint[0, :] = np.ones((1, pos_edges))
            # constraint_mat = np.vstack([constraint_mat, edge_constraint])
            b_l = np.hstack([np.zeros(constraint_mat.shape[0] - 1), [0]]) #, [np.sum(sol)]])
            b_u0 = b_u
            b_u = np.hstack([b_u0, [np.sum(sol)-1]]) #, [np.sum(sol)]])
              # Upper bound allows overlap
    else:
        b_l = np.zeros(constraint_mat.shape[0])
        b_u = np.hstack((X[nodes,:]@capacity - np.array([initial_graph.degree[i] for i in nodes]),(X[nodes,:]@O - nx.adjacency_matrix(initial_graph).todense()@X[nodes,:]).flatten()))
    # Create the solution coefficients
    c = -np.ones(pos_edges - len(edges_idx))
    integrality = np.ones_like(c)
    # Create the linear constraint
    constraints = LinearConstraint(constraint_mat, b_l, b_u)
    # Solve the linear programming problem
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=(0,1), options={'disp':disp, 'time_limit': 120})
    if res.success:
        if solution:
            if ret_edges:
                added_edges = []
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        added_edges.append((nodes[i], nodes[j]))
                added_edges = np.array([e for e in added_edges if e not in initial_graph.edges()])
                return res.x, added_edges[res.x.astype(int) == 1,:]
            return res.x
        else:
            if ret_edges:
                added_edges = []
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        added_edges.append((nodes[i], nodes[j]))
                added_edges = np.array([e for e in added_edges if e not in initial_graph.edges()])
                return -int(res.fun), added_edges[res.x.astype(int) == 1,:]
            return -int(res.fun)
    else:
        if solution:
            if ret_edges:
                return None, None
            return None
        if ret_edges:
            return None, None
        return None

def self_assembly(X,O,capacity,initial_graph):
    """
    Determine whether network can self-assemble.
    
    Parameters:
        X (ndarray): Matrix of node labels.
        O (ndarray): Binding matrix.
        capacity (ndarray): Capacity vector.
        initial_graph (networkx.Graph): Initial graph.
    Returns:
        bool: True if the network can self-assemble, False otherwise.
    """
    sol, edges = find_optimal_edge_count(X, O, capacity, initial_graph=initial_graph, solution=True, ret_edges=True)
    if sol is None:
        return False
    solutions = [sol]
    edge_lists = [edges]
    # Check whether another solution exists
    new_sol = True
    while new_sol is not None:
        new_sol, new_edges = find_optimal_edge_count(X, O, capacity, old_sol=solutions, initial_graph=initial_graph, solution=True, ret_edges=True)
        # Check whether new_sol has right amount of edges
        if len(new_edges) < len(edge_lists[0]):
            break

        g = nx.Graph()
        g.add_nodes_from(initial_graph.nodes())
        g.add_edges_from(new_edges)
        # Check whether isomorphic
        for i in range(len(edge_lists)):
            h = nx.Graph()
            h.add_nodes_from(initial_graph.nodes())
            h.add_edges_from(edge_lists[i])
            if not nx.is_isomorphic(g,h):
                return False
        solutions.append(new_sol)
        edge_lists.append(new_edges)
    return True
    
def rewire(g,X,O,capacity,T,burn_in=100,fixed_edges=None,sample=True):
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
    if sample:
        # Get number of edges
        E = g.number_of_edges()
        new_g = nx.Graph()
        new_g.add_nodes_from(g.nodes())
        if fixed_edges is not None:
            new_g.add_edges_from(fixed_edges)
        # Order all possible edges
        pos_edges = list(combinations(g.nodes(), 2))
        # Random shuffle edges
        idx = np.arange(len(pos_edges))
        np.random.shuffle(idx)
        i = 0
        #print("----")
        #O = O.astype(int)
        #capacity = capacity.astype(int)
        #X = X.astype(int)
        while new_g.number_of_edges() < E:
            # Try and add edge
            e1 = pos_edges[idx[i]]
            v1, v2 = e1
            #print(v1,v2)
            t1, t2 = X[v1].argmax(), X[v2].argmax()
            if new_g.has_edge(v1,v2) or new_g.degree(v1) == capacity[t1] or new_g.degree(v2) == capacity[t2]:
                i += 1
                #continue
            # Check whether edge is compatible
            #elif new_g.degree(v1) == capacity[t1] or new_g.degree(v2) == capacity[t2]:
            #    i += 1
                #continue
            #
            # Get number of connections with each type
            else:
                v1_neighbors = list(new_g.neighbors(v1))
                v2_neighbors = list(new_g.neighbors(v2))
                v1_types = X[v1_neighbors].sum(axis=0)
                v2_types = X[v2_neighbors].sum(axis=0)
                #print(v1_types,v2_types)
                #print(O[t1,:],O[t2,:])
                #print(O[t1,t2],v1_types[t2],O[t2,t1],v2_types[t1])
                if O[t1, t2] > v1_types[t2] and O[t2, t1] > v2_types[t1]:
                    new_g.add_edge(v1, v2)
                    #print("added")
                    #print(v1,v2)
                i += 1
            #print(v1,v2)
            #print(i)
            if i >= len(idx) and new_g.number_of_edges() < E:
                #print("broke")
                i = 0
                new_g.remove_edges_from(list(new_g.edges()))
                if fixed_edges is not None:
                    # Re-add fixed edges
                    new_g.add_edges_from(fixed_edges)
                np.random.shuffle(idx)
        return new_g
            

    if not sample:
        nodes = list(g.nodes())
        node_capacity_per_type = X @ O
        node_capacity = X @ capacity[:,np.newaxis]
        edges = list(g.edges())
        # Remove fixed edges from edge list
        if fixed_edges is not None:
            available_edges = [e for e in edges if e not in fixed_edges and e[::-1] not in fixed_edges]
        else:
            available_edges = edges.copy()
            fixed_edges = []
        
        # Burn in period
        for _ in range(burn_in):
            # Randomly select edge to rewire
            edges = list(g.edges())
            if fixed_edges is not None:
                available_edges = [e for e in edges if e not in fixed_edges and e[::-1] not in fixed_edges]
            else:
                available_edges = edges.copy()
                fixed_edges = []
            if len(available_edges) == 0:
                break
            e1 = random.choice(available_edges)
            e1 = np.array(e1)
            # Randomly order nodes of edge
            np.random.shuffle(e1)
            e1 = tuple(e1)
            v1, v2 = e1
            # Get types of nodes
            t1, t2 = X[v1].argmax(), X[v2].argmax()
            # Randomly choose two nodes, without replacement
            v3, v4 = np.random.choice(nodes, 2, replace=False)
            # Check whether edge was chosen
            if (v3,v4) == e1 or (v4,v3) == e1 or (v3,v4) in fixed_edges or (v4,v3) in fixed_edges:
                continue
            # Get types of new nodes
            t3, t4 = X[v3].argmax(), X[v4].argmax()
            # Check whether nodes are connected
            if g.has_edge(v3,v4) and v3 != v1 and v3 != v2 and v4 != v1 and v4 != v2:
                if g.has_edge(v1,v3) or g.has_edge(v2,v4):
                    continue
                # Check whether swap is compatible
                if node_capacity_per_type[v1, t3] > 0 and node_capacity_per_type[v2, t4] > 0 and node_capacity_per_type[v3, t1] > 0 and node_capacity_per_type[v4, t2] > 0:
                    # Remove old edge and add new edge
                    g.remove_edge(v1, v2)
                    g.remove_edge(v3, v4)
                    g.add_edge(v1, v3)
                    g.add_edge(v2, v4)
                    
                    # Update available edges
                    # if (v1,v2) in available_edges:
                    #     available_edges.remove((v1,v2))
                    # else:
                    #     available_edges.remove((v2,v1))
                    # if (v3,v4) in available_edges:
                    #     available_edges.remove((v3,v4))
                    # else:
                    #     available_edges.remove((v4,v3))
                    # available_edges.append((v1, v3))
                    # available_edges.append((v2,v4))
            elif not g.has_edge(v3,v4):
                # Check whether v3 and v4 are at capacity
                if g.degree(v3) == node_capacity[v3] and g.degree(v4) == node_capacity[v4]:
                    continue
                if v3 == v1:
                    if g.has_edge(v2, v4):
                        continue
                    # Check whether v4 can connect to v2 
                    if g.degree(v4) == node_capacity[v4] or node_capacity_per_type[v4, t2] == 0:
                        continue
                    else:
                        # Check whether v4 is of type t1
                        if t4 == t1:
                            g.remove_edge(v1, v2)
                            g.add_edge(v2, v4)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v2, v4))
                        else:
                            # Check whether v2 can connect to v4
                            if node_capacity_per_type[v2,t4] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v2, v4)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                                # available_edges.append((v2, v4))
                elif v3 == v2:
                    if g.has_edge(v1, v4):
                        continue
                    # Check whether v4 can connect to v1 
                    if g.degree(v4) == node_capacity[v4] or node_capacity_per_type[v4, t1] == 0:
                        continue
                    else:
                        # Check whether v4 is of type t2
                        if t4 == t2:
                            g.remove_edge(v1, v2)
                            g.add_edge(v1, v4)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v1, v4))
                        else:
                            # Check whether v1 can connect to v4
                            if node_capacity_per_type[v1,t4] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v1, v4)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                        
                                # available_edges.append((v1, v4))
                    
                elif v4 == v1:
                    if g.has_edge(v2, v3):
                        continue
                    # Check whether v3 can connect to v2 
                    if g.degree(v3) == node_capacity[v3] or node_capacity_per_type[v3, t2] == 0:
                        continue
                    else:
                        # Check whether v3 is of type t1
                        if t3 == t1:
                            g.remove_edge(v1, v2)
                            g.add_edge(v2, v3)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v2, v3))
                        else:
                            # Check whether v2 can connect to v3
                            if node_capacity_per_type[v2,t3] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v2, v3)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                                # available_edges.append((v2, v3))
                    
                elif v4 == v2:
                    if g.has_edge(v1, v3):
                        continue
                    # Check whether v3 can connect to v1 
                    if g.degree(v3) == node_capacity[v3] or node_capacity_per_type[v3, t1] == 0:
                        continue
                    else:
                        # Check whether v3 is of type t2
                        if t3 == t2:
                            g.remove_edge(v1, v2)
                            g.add_edge(v1, v3)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v1, v3))
                        else:
                            # Check whether v1 can connect to v3
                            if node_capacity_per_type[v1,t3] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v1, v3)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                                # available_edges.append((v1, v3))
                    
                elif v1 != v3 and v1 != v4 and v2 != v3 and v2 != v4:
                    # If none of the nodes are the same, try to make one new connection
                    # Check which nodes are compatible
                    pos_edges = []
                    for n1 in [v1,v2]:
                        for n2 in [v3,v4]:
                            if g.degree(n2) == node_capacity[n2]:
                                continue
                            else:
                                if node_capacity_per_type[n2, X[n1].argmax()] > 0 and node_capacity_per_type[n1, X[n2].argmax()] > 0:
                                    if not g.has_edge(n1, n2):
                                        # Only add edge if it is not already present
                                        pos_edges.append((n1,n2))
                    # Randomly select new edge
                    if len(pos_edges) > 0:
                        new_edge = random.choice(pos_edges)
                        g.remove_edge(v1, v2)
                        g.add_edge(*new_edge)
                        # Update available edges
                        # if (v1,v2) in available_edges:
                        #     available_edges.remove((v1, v2))
                        # else:
                        #     available_edges.remove((v2, v1))
                        # available_edges.append(new_edge)
        # After burn in perform for T iterations
        for _ in range(T):
            # Randomly select edge to rewire
            edges = list(g.edges())
            if fixed_edges is not None:
                available_edges = [e for e in edges if e not in fixed_edges and e[::-1] not in fixed_edges]
            else:
                available_edges = edges.copy()
                fixed_edges = []
            if len(available_edges) == 0:
                break
            e1 = random.choice(available_edges)
            e1 = np.array(e1)
            # Randomly order nodes of edge
            np.random.shuffle(e1)
            e1 = tuple(e1)
            v1, v2 = e1
            # Get types of nodes
            t1, t2 = X[v1].argmax(), X[v2].argmax()
            # Randomly choose two nodes, without replacement
            v3, v4 = np.random.choice(nodes, 2, replace=False)
            # Check whether edge was chosen
            if (v3,v4) == e1 or (v4,v3) == e1 or (v3,v4) in fixed_edges or (v4,v3) in fixed_edges:
                continue
            # Get types of new nodes
            t3, t4 = X[v3].argmax(), X[v4].argmax()
            # Check whether nodes are connected
            if g.has_edge(v3,v4) and v3 != v1 and v3 != v2 and v4 != v1 and v4 != v2:
                if g.has_edge(v1,v3) or g.has_edge(v2,v4):
                    continue
                # Check whether swap is compatible
                if node_capacity_per_type[v1, t3] > 0 and node_capacity_per_type[v2, t4] > 0 and node_capacity_per_type[v3, t1] > 0 and node_capacity_per_type[v4, t2] > 0:
                    # Remove old edge and add new edge
                    g.remove_edge(v1, v2)
                    g.remove_edge(v3, v4)
                    g.add_edge(v1, v3)
                    g.add_edge(v2, v4)
                    
                    # Update available edges
                    # if (v1,v2) in available_edges:
                    #     available_edges.remove((v1,v2))
                    # else:
                    #     available_edges.remove((v2,v1))
                    # if (v3,v4) in available_edges:
                    #     available_edges.remove((v3,v4))
                    # else:
                    #     available_edges.remove((v4,v3))
                    # available_edges.append((v1, v3))
                    # available_edges.append((v2,v4))
            elif not g.has_edge(v3,v4):
                # Check whether v3 and v4 are at capacity
                if g.degree(v3) == node_capacity[v3] and g.degree(v4) == node_capacity[v4]:
                    continue
                if v3 == v1:
                    if g.has_edge(v2, v4):
                        continue
                    # Check whether v4 can connect to v2 
                    if g.degree(v4) == node_capacity[v4] or node_capacity_per_type[v4, t2] == 0:
                        continue
                    else:
                        # Check whether v4 is of type t1
                        if t4 == t1:
                            g.remove_edge(v1, v2)
                            g.add_edge(v2, v4)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v2, v4))
                        else:
                            # Check whether v2 can connect to v4
                            if node_capacity_per_type[v2,t4] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v2, v4)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                                # available_edges.append((v2, v4))
                elif v3 == v2:
                    if g.has_edge(v1, v4):
                        continue
                    # Check whether v4 can connect to v1 
                    if g.degree(v4) == node_capacity[v4] or node_capacity_per_type[v4, t1] == 0:
                        continue
                    else:
                        # Check whether v4 is of type t2
                        if t4 == t2:
                            g.remove_edge(v1, v2)
                            g.add_edge(v1, v4)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v1, v4))
                        else:
                            # Check whether v1 can connect to v4
                            if node_capacity_per_type[v1,t4] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v1, v4)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                        
                                # available_edges.append((v1, v4))
                    
                elif v4 == v1:
                    if g.has_edge(v2, v3):
                        continue
                    # Check whether v3 can connect to v2 
                    if g.degree(v3) == node_capacity[v3] or node_capacity_per_type[v3, t2] == 0:
                        continue
                    else:
                        # Check whether v3 is of type t1
                        if t3 == t1:
                            g.remove_edge(v1, v2)
                            g.add_edge(v2, v3)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v2, v3))
                        else:
                            # Check whether v2 can connect to v3
                            if node_capacity_per_type[v2,t3] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v2, v3)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                                # available_edges.append((v2, v3))
                    
                elif v4 == v2:
                    if g.has_edge(v1, v3):
                        continue
                    # Check whether v3 can connect to v1 
                    if g.degree(v3) == node_capacity[v3] or node_capacity_per_type[v3, t1] == 0:
                        continue
                    else:
                        # Check whether v3 is of type t2
                        if t3 == t2:
                            g.remove_edge(v1, v2)
                            g.add_edge(v1, v3)
                            # Update available edges
                            # if (v1,v2) in available_edges:
                            #     available_edges.remove((v1, v2))
                            # else:
                            #     available_edges.remove((v2, v1))
                            # available_edges.append((v1, v3))
                        else:
                            # Check whether v1 can connect to v3
                            if node_capacity_per_type[v1,t3] > 0:
                                g.remove_edge(v1, v2)
                                g.add_edge(v1, v3)
                                # Update available edges
                                # if (v1,v2) in available_edges:
                                #     available_edges.remove((v1, v2))
                                # else:
                                #     available_edges.remove((v2, v1))
                                # available_edges.append((v1, v3))
                    
                elif v1 != v3 and v1 != v4 and v2 != v3 and v2 != v4:
                    # If none of the nodes are the same, try to make one new connection
                    # Check which nodes are compatible
                    pos_edges = []
                    for n1 in [v1,v2]:
                        for n2 in [v3,v4]:
                            if g.degree(n2) == node_capacity[n2]:
                                continue
                            else:
                                if node_capacity_per_type[n2, X[n1].argmax()] > 0 and node_capacity_per_type[n1, X[n2].argmax()] > 0:
                                    if not g.has_edge(n1, n2):
                                        # Only add edge if it is not already present
                                        pos_edges.append((n1,n2))
                    # Randomly select new edge
                    if len(pos_edges) > 0:
                        new_edge = random.choice(pos_edges)
                        g.remove_edge(v1, v2)
                        g.add_edge(*new_edge)
                        # Update available edges
                        # if (v1,v2) in available_edges:
                        #     available_edges.remove((v1, v2))
                        # else:
                        #     available_edges.remove((v2, v1))
                        # available_edges.append(new_edge)
        return g















    # n_edges = g.number_of_edges()
    
    # nodes = list(g.nodes())


    # v_to_ix = {v:i for i,v in enumerate(g.nodes())}
    # ix_to_v = {i:v for i,v in enumerate(g.nodes())}
    # #An NxN binary matrix M where M_ij = 1 iff the O matrix allows an edge between nodes i and j
    # node_capacity_per_type = X[nodes] @ O 
    # node_capacity = X[nodes] @ capacity[:,np.newaxis]

    # g = g.remove_edges_from(g.edges())

    # for e in fixed_edges:
    #     g.add_edge(*e)
    #     node_capacity[v_to_ix[e[0]]]-=1
    #     node_capacity[v_to_ix[e[1]]]-=1
    #     node_capacity_per_type[v_to_ix[e[0]],X[e[1]].argmax()]-=1
    #     node_capacity_per_type[v_to_ix[e[1]],X[e[0]].argmax()]-=1

    




    

    '''
    A = nx.adjacency_matrix(g).todense()
    nodes = list(g.nodes())

    v_to_ix = {v:i for i,v in enumerate(g.nodes())}
    ix_to_v = {i:v for i,v in enumerate(g.nodes())}
    #An NxN binary matrix M where M_ij = 1 iff the O matrix allows an edge between nodes i and j
    pos_edgesO = X[nodes] @ O @ X[nodes].T
    
    AXX = A@X[nodes]@X[nodes].T
    
    #Set entries to zero if they correspond to self loops or existing edges
    pos_edgesO -= AXX

    pos_edgesO = (pos_edgesO > 0).astype(int)
    pos_edgesO -= np.diag(np.diag(pos_edgesO))
    
    #An Nx1 vector v with the unused capacity of the nodes
    free_sitesC = X[nodes] @ capacity[:,np.newaxis] - np.array([g.degree(v) for v in g.nodes()],dtype=int)[:,np.newaxis]
    #print(freesites.shape,(X[nodes] @ capacity),np.array([g.degree(v) for v in g.nodes()],dtype=int),freesites)
    
    free_sitesO = X[nodes] @ O - A @ X[nodes]

    #An Nx1 vector v where v_i = 0 iff node i is at full capacity and all its incident edges are fixed
    full_nodes_with_only_fixed_edges = np.array([0 if free_sitesC[v_to_ix[v]] == 0 and np.all([1 if e in fixed_edges else 0 for e in g.edges(v)]) else 1 for v in g.nodes()])[:, np.newaxis]

    #An NxN binary matrix M where M_ij = 1 iff at least one of the two nodes i or j has free sites. 
    
    pos_edgesC = np.full((g.number_of_nodes(),g.number_of_nodes()),1) - (free_sitesC == 0).astype(int) @ ((free_sitesC == 0).astype(int)).T
    pos_edgesC -= np.diag(np.diag(pos_edgesC))
    
    #Set those entries of pos_edgesC to zero where at least one of the two nodes is full AND has only fixed edges incident to it
    pos_edgesC *= full_nodes_with_only_fixed_edges @ full_nodes_with_only_fixed_edges.T

    pos_edgesO = np.full((g.number_of_nodes(),g.number_of_nodes()),1) - (free_sitesO == 0).astype(int) @ (free_sitesO == 0).astype(int).T
    pos_edgesO -= np.diag(np.diag(pos_edgesO))
    
    #If all edges in the graph are fixed then dont rewire
    if np.all([1 if e in fixed_edges else 0 for e in g.edges()]):
        return g
    
    
    for i in range(T):
        #Choose an edge to add from the possible ones, i.e., those edges ij where pos_edgesC_ij=pos_edgesO_ij==1
        possible_edges = np.nonzero(np.logical_and(pos_edgesC,pos_edgesO))
        rows, cols = possible_edges

        indices = [(int(r), int(c)) for r, c in zip(rows, cols)]

        #If no such edges exist return g
        if not indices:
            return g
        else:
            e2 = random.choice(indices)

        edgelist = list(g.edges)
        print(e2)
        print(ix_to_v[e2[0]],ix_to_v[e2[1]])
        
        #If both edge points of the new edge have free sites choose a non fixed edge from the set of all edges in gto remove
        if freesites[e2[0]] > 0 and freesites[e2[1]] > 0:
            succes = False
            while not succes:
                e1 = random.choice(edgelist)
                if e1 not in fixed_edges:
                    succes = True

        #If the first end point of the new edge has no free sites choose a non fixed edge from the set of edges incident to that end point to remove
        elif freesites[e2[0]] == 0 and freesites[e2[1]] > 0:
            succes = False
            edgesv = list(g.edges(ix_to_v[e2[0]]))
            
            while not succes:
                e1 = random.choice(edgesv)
                if e1 not in fixed_edges:
                    succes = True

        #If the second end point of the new edge has no free sites choose a non fixed edge from the set of edges incident to that end point to remove
        elif freesites[e2[0]] > 0 and freesites[e2[1]] == 0:
            succes = False
            edgesv = list(g.edges(ix_to_v[e2[1]]))
            
            while not succes:
                e1 = random.choice(edgesv)
                if e1 not in fixed_edges:
                    succes = True
        
        else:
            print("error")

        #Remove the old edge and add the new
        g.remove_edge(*e1)
        g.add_edge(ix_to_v[e2[0]],ix_to_v[e2[1]])


        ix10 = v_to_ix[e1[0]]
        ix11 = v_to_ix[e1[1]]

        #Update the freesites vector
        freesites[e2[0]] += 1
        freesites[e2[1]] += 1
        freesites[ix10] -= 1
        freesites[ix11] -= 1

        #Update the pos_edgesC matrix
        if freesites[e2[0]] > 0  or freesites[e2[1]] > 0:
            pos_edgesC[e2[0],e2[1]] = 1
            pos_edgesC[e2[1],e2[0]] = 1
        else:
            pos_edgesC[e2[0],e2[1]] = 0
            pos_edgesC[e2[1],e2[0]] = 0

        if freesites[ix10] > 0 or freesites[ix11] > 0:
            pos_edgesC[ix10,ix11] = 1
            pos_edgesC[ix11,ix10] = 1
        else:
            pos_edgesC[ix10,ix11] = 0
            pos_edgesC[ix11,ix10] = 0

        #Update the pos_edgesO matrix
        pos_edgesO[ix10,ix11] +=1
        pos_edgesO[ix11,ix10] +=1
        pos_edgesO[e2[0],e2[1]] -=1
        pos_edgesO[e2[1],e2[0]] -=1

    
    return g
    
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
        if (u,u_neighbor) in fixed_edges or (u_neighbor,u) in fixed_edges or v == u_neighbor:
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
            if w == u:
                continue
            # Update pos_neighbors
            w_label = X[w].argmax()
            if (v,w) in fixed_edges or (w,v) in fixed_edges:
                continue
            # Check whether edges already exist
            if (v,u_neighbor) in g.edges() or (u_neighbor,v) in g.edges() or (u,w) in g.edges() or (w,u) in g.edges():
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
    '''
    # return g

def prob_dist(X,O,capacity,max_iters=100,initial_graph=None,multiedge=False,verbose=False,labeled=False,T=1000,max_edges=False, rewire_est=True):
    """
    Extract empirical distribution of system.
    
    Parameters:
        X (ndarray) - matrix of node labels
        O (ndarray) - binding matrix
        
    Returns:
        int - stability index
    """
    success = True
    cur_graphs = []
    if initial_graph is not None and initial_graph.number_of_nodes() == 1:
        return np.array([max_iters]), [initial_graph], np.array([0]), True
    if verbose:
        for t in tqdm(range(max_iters)):
            if not rewire_est:
                test_g, rates = microcanonical_ensemble(X,O,capacity,T=T,initial_graph=initial_graph.copy(),multiedge=multiedge,kappa_d=.1,ret_rates = True,max_edges=max_edges)
                if rates[:-test_g.number_of_nodes()].sum() != 0 and max_edges is False:
                    continue
                cur_graphs.append(test_g.copy())
            else:
                if t == 0:
                    A_flat, edges = find_optimal_edge_count(X,O,capacity,initial_graph=initial_graph.copy(),solution=False,disp=False,ret_edges=True)
                    if A_flat is None:
                        success = False
                        test_g, rates = microcanonical_ensemble(X,O,capacity,T=1000,initial_graph=initial_graph.copy(),multiedge=multiedge,kappa_d=.1,ret_rates = True,max_edges=max_edges)
                        if rates[:-test_g.number_of_nodes()].sum() != 0 and max_edges is False:
                            continue
                    else:
                        test_g = initial_graph.copy()
                        test_g.add_edges_from(edges)
                    cur_graphs.append(test_g.copy())
                else:
                    # cur_graphs.append(test_g.copy())
                    test_g = rewire(test_g.copy(),X,O,capacity,T=int(2*test_g.number_of_edges()),fixed_edges=list(initial_graph.edges()))
                    cur_graphs.append(test_g.copy())
    else:
        for t in range(max_iters):
            if not rewire_est:
                test_g, rates = microcanonical_ensemble(X,O,capacity,T=T,initial_graph=initial_graph.copy(),multiedge=multiedge,kappa_d=.1,ret_rates = True,max_edges=max_edges)
                if rates[:-test_g.number_of_nodes()].sum() != 0 and max_edges is False:
                    continue
                cur_graphs.append(test_g.copy())
            else:
                if t == 0:
                    A_flat, edges = find_optimal_edge_count(X,O,capacity,initial_graph=initial_graph.copy(),solution=False,disp=False,ret_edges=True)
                    if A_flat is None:
                        success = False
                        test_g, rates = microcanonical_ensemble(X,O,capacity,T=1000,initial_graph=initial_graph.copy(),multiedge=multiedge,kappa_d=.1,ret_rates = True,max_edges=max_edges)
                        if rates[:-test_g.number_of_nodes()].sum() != 0 and max_edges is False:
                            continue
                    else:
                        test_g = initial_graph.copy()
                        test_g.add_edges_from(edges)
                    cur_graphs.append(test_g.copy())
                else:
                    # cur_graphs.append(test_g.copy())
                    test_g = rewire(test_g.copy(),X,O,capacity,T=int(2*test_g.number_of_edges()),fixed_edges=list(initial_graph.edges()))
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
    return np.array(counts), final_graphs, sorted_indices, success

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
    kappa_d=0.1,
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
    rates_detach = kappa_d * np.array([1 - g.degree(j) / capacity[labels[i]] for i, j in enumerate(g.nodes())])
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

            rates_detach[idx_i] = kappa_d * (1 - g.degree(i) / capacity[i_label])
            rates_detach[idx_j] = kappa_d * (1 - g.degree(j) / capacity[j_label])
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
                rates_detach[idx_j] = kappa_d * (1 - g.degree(j) / capacity[labels[idx_j]])

            missing_edges = [edge for edge in true_edges if not g.has_edge(*edge)]
            g.add_edges_from(missing_edges)
            # Get current potential links
            potential_links = (X@O - nx.adjacency_matrix(g).todense()@X)@X.T
            # Update compatibility
            compatibility = np.heaviside(potential_links,0.0).astype(int)
            # Update rates
            rates_attach = compatibility[np.triu_indices(N)] * compatibility.T[np.triu_indices(N)] * kappa_a
            rates_detach[idx_i] = kappa_d * (1 - g.degree(i) / capacity[i_label])
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
    # X = np.vstack([np.eye(3) for i in range(2)])
    # O = np.array([[0,1,1],[1,0,1],[1,1,2]])
    # capacity = O.sum(axis=1,dtype=int)
    # target = nx.Graph()
    # target.add_nodes_from(np.arange(6))
    # target.add_edges_from([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[2,5]])
    O = np.array([[0,1,1],[1,0,1],[1,1,0]],dtype=int)
    X = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]],dtype=int)
    capacity = O.sum(axis=1,dtype=int)
    g = nx.Graph()
    g.add_nodes_from(np.arange(4))
    g.add_edges_from([(0,1),(1,2),(2,0)])
    print(g.edges())
    # new_g = microcanonical_ensemble(X,O,capacity)
    # draw_network(new_g,X,with_labels=True)
    # for i in range(10):
    start = time.time()
    for i in range(10):
        new_g = rewire(g,X,O,capacity,T=1000,sample=True)
        print(new_g.edges())
    print('sample',time.time() - start)
    start = time.time()
    for i in range(10):
        new_g = rewire(g,X,O,capacity,T=1000,sample=False)
        print(new_g.edges())
    print('rewire',time.time() - start)
    draw_network(new_g,X,with_labels=True)
