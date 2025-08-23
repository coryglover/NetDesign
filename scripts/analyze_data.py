import numpy as np 
import networkx as nx
import os
import pandas as pd 
import assembly_tree as at 
import mcmc
import argparse
import json 
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze graph data and compute assembly trees.')
    parser.add_argument('--graph_file', type=str, required=True, help='Directory containing input files.')
    parser.add_argument('--X_file', type=str, required=True, help='File containing node labels.')
    parser.add_argument('--tree_file', type=str, required=True, help='File containing the assembly tree.')
    parser.add_argument('--output', type=str, required=True, help='File to save results.')
    parser.add_argument("--c_exclusive",action="store_true",default=False)
    parser.add_argument("--O_file",type=str, default=None, help='Path to O file')
    parser.add_argument("--c_file",type=str, default=None,help='Path to c file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if dataframe exists
    if os.path.exists(args.output):
        df = pd.read_csv(args.output, index_col=0)
    else:
        # Create a new dataframe with the specified columns
        df = pd.DataFrame(columns=[
            'name', 'subdir', 'N', 'E', 'N_types', 'specificity',
            'degree_mean', 'degree_hetero', 'max_edges',
            'clustering_coeff', 'self_assembly_probability',
            'tree_num','guided_assembly_probability',
            'tree_depth','tree_leaves','tree_N'
        ])

    # Load graph
    target = nx.read_edgelist(args.graph_file, nodetype=int, create_using=nx.Graph)
    print(f"Loaded graph with {target.number_of_nodes()} nodes and {target.number_of_edges()} edges.")
    # Load labels
    X = np.loadtxt(args.X_file, dtype=int)
    if X.ndim == 1:
        X = X.reshape(len(X),1)
    # Get capacity vector
    capacity = at.extract_deg_cap(target, X).reshape(-1)
    if args.c_file is not None:
        capacity = np.loadtxt(args.c_file)
    else:
        capacity = at.extract_deg_cap(target, X).reshape(-1)
    if args.O_file is not None:
        O = np.loadtxt(args.O_file)
    else:
        if args.c_exclusive:
            O = (np.ones((X.shape[1],X.shape[1])) * capacity).T
        else:
            O = at.extract_O(target, X)

    # Try to load tree
    trees = []
    try:
        with open(args.tree_file, 'r') as f:
            tree_data = json.load(f)
        for tree_dict in tree_data:
            T = mcmc.AssemblyTree(target, X, O, capacity, multiedge=False)
            T = T.load_tree(tree_dict, T)
            trees.append(T)
    except:
        trees = []

    # Calculate network and design set statistics
    N = target.number_of_nodes()
    E = target.number_of_edges()
    N_types = X.shape[1]

    # Get specificity
    O_sum = np.sum(O,axis=1)
    total_psi = 0
    for i in range(N):
        label = np.argmax(X[i])
        psi_i = O_sum[label] - capacity[label]
        psi_i /= (capacity[label]*N_types - capacity[label])
        total_psi += psi_i
    total_psi /= N
    total_psi = 1 - total_psi

    # Get degree stats
    degree_sequence = [d for n, d in target.degree()]
    degree_mean = np.mean(degree_sequence)
    degree_hetero = np.mean([d**2 for d in degree_sequence])

    # Max_edges
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from(np.arange(N))
    _, opt_edges = at.find_optimal_edge_count(X, O, capacity, initial_graph=initial_graph,solution=False,disp=False,ret_edges=True)
    max_edges = len(opt_edges)

    # Get clustering coefficient
    clustering_coeff = nx.average_clustering(target)
    if isinstance(clustering_coeff, dict):
        clustering_coeff = np.mean(list(clustering_coeff.values()))
    
    # Get self assembly probability
    if max_edges != E:
        sa_p = 0
    else:
        initial_graph = nx.Graph()
        initial_graph.add_nodes_from(np.arange(N))
        for k in range(2,100):
            p, samples, idx, success = at.prob_dist(X, O, capacity, initial_graph=initial_graph, max_iters=k,max_edges=True,rewire_est=False)
            if len(p) > 1:
                break

        if len(p) == 1:
            if nx.is_isomorphic(target, samples[0]):
                sa_p = 1
        else:
            sa_p = 0
    
    # Get name information
    graph_name = args.graph_file.split('/')[-1]
    name = graph_name.split('.')[0]
    subdir = args.graph_file.split('/')[6]

    graph_stats = {
        'name': [name],
        'subdir': [subdir],
        'N': [N],
        'E': [E],
        'N_types': [N_types],
        'specificity': [total_psi],
        'degree_mean': [degree_mean],
        'degree_hetero': [degree_hetero],
        'max_edges': [max_edges],
        'clustering_coeff': [clustering_coeff],
        'self_assembly_probability': [sa_p]
    }

    if len(trees) == 0:
        stats = copy.deepcopy(graph_stats)
        stats['tree_num'] = [np.nan]
        stats['guided_assembly_probability'] = [np.nan]
        stats['tree_depth'] = [np.nan]
        stats['tree_leaves'] = [np.nan]
        stats['tree_N'] = [np.nan]

        # Append the stats to the dataframe
        df = pd.concat([df,pd.DataFrame(stats)], ignore_index=True)
        # Save the dataframe to the output file
        df.to_csv(args.output, index=False)
    else:
        for i, T in enumerate(trees):
            stats = copy.deepcopy(graph_stats)
            stats[f'tree_num'] = [i]
            stats[f'guided_assembly_probability'] = [np.sum(T.Tree.get_node(0).data.p)]
            stats[f'tree_depth'] = [T.Tree.depth()]
            stats[f'tree_leaves'] = [len(T.Tree.leaves())]
            stats[f'tree_N'] = [T.Tree.all_nodes()]

            # Append the stats to the dataframe
            df = pd.concat([df,pd.DataFrame(stats)], ignore_index=True)
            # Save the dataframe to the output file
            df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
    

    

    
