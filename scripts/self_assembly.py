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

    initial_graph = nx.Graph()
    initial_graph.add_nodes_from(np.arange(X.shape[0]))
    self_assembly = at.self_assembly(X, O, capacity, initial_graph)


    # Add value to next line of file
    # Make new output file
    output_path = args.output.split('/')[:-1]
    output_path = '/'.join(output_path)
    output = f'{output_path}/self_assembly.txt'
    with open(output, 'a') as f:
        f.write(f"{self_assembly}\n")

if __name__ == '__main__':
    main()
