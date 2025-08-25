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
    parser.add_argument('--param_file', type=str, required=True, help='Directory containing input files.')
    parser.add_argument('--output', type=str, required=True, help='File to save results.')
    parser.add_argument("--c_exclusive",action="store_true",default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    # Get graphs
    with open(args.param_file, 'r') as f:
        lines = f.readlines()
    # for line in lines:
    #     line.split(' ')
    graph_files = [line.split('--graph_file ')[1].split(' --X_file')[0] for line in lines]
    X_files = [line.split('--X_file ')[1].split(' --')[0] for line in lines]
    O_file = [line.split('--O_file ')[1].split(' --')[0] if '--O_file' in line else None for line in lines]
    c_file = [line.split('--c_file ')[1].split(' --')[0] if '--c_file' in line else None for line in lines]
    
    sa_p = np.zeros(len(graph_files))
    for i in range(len(graph_files)):

        # Load graph
        target = nx.read_edgelist(graph_files[i], nodetype=int, create_using=nx.Graph)
        print(f"Loaded graph with {target.number_of_nodes()} nodes and {target.number_of_edges()} edges.")
        # Load labels
        X = np.loadtxt(X_files[i], dtype=int)
        if X.ndim == 1:
            X = X.reshape(len(X),1)
        # Get capacity vector
        capacity = at.extract_deg_cap(target, X).reshape(-1)
        if args.c_file is not None:
            capacity = np.loadtxt(c_file[i])
        else:
            capacity = at.extract_deg_cap(target, X).reshape(-1)
        if args.O_file is not None:
            O = np.loadtxt(O_file[i])
        else:
            if args.c_exclusive:
                O = (np.ones((X.shape[1],X.shape[1])) * capacity).T
            else:
                O = at.extract_O(target, X)

        initial_graph = nx.Graph()
        initial_graph.add_nodes_from(np.arange(X.shape[0]))
        sa_p[i] = at.self_assembly(X, O, capacity, initial_graph)
    
    np.savetxt(args.output, sa_p)

if __name__ == "__main__":
    main()
