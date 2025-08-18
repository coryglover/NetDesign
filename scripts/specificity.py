import numpy as np
import assembly_tree as at
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import sympy
import os
from scipy.special import stirling2
import networkx as nx
import treelib
from itertools import product
import copy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run MCMC to identify best trees for datasets.")
    parser.add_argument("--graph_file", type=str, required=True, help='Path to specific network')
    parser.add_argument("--X_file", type=str, required=True, help="Path to the X matrix file.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to draw in MCMC.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results.")
    parser.add_argument("--c_exclusive",action="store_true",default=False)
    parser.add_argument("--multiedge", type=int, required=False, default = 0, help="1 if multiedge")
    parser.add_argument("--MAPonly", type=int, required=False, default = 1, help="Only care about the MAP")
    parser.add_argument("--O_file",type=str, default=None, help='Path to O file')
    parser.add_argument("--c_file",type=str, default=None,help='Path to c file')
    return parser.parse_args()

def main():
    args = parse_args()
    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Graph name
    graph_name = os.path.splitext(os.path.basename(args.graph_file))[0]

    # Read in target graph
    target = nx.read_edgelist(args.graph_file, nodetype=int)
    # Read in X matrix
    X = np.loadtxt(args.X_file, delimiter=' ')
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

    O_sum = np.sum(O,axis=1)
    N_types = X.shape[1]
    N = X.shape[0]

    total_psi = 0
    for i in range(N):
        label = np.argmax(X[i])
        psi_i = O_sum[label] - capacity[label]
        psi_i /= (capacity[label]*N_types - capacity[label])
        total_psi += psi_i
    total_psi /= N
    total_psi = 1 - total_psi

    with open(args.output+'specificity.txt','a') as f:
        f.write(f'{total_psi}\n')

if __name__ == '__main__':
    main()
