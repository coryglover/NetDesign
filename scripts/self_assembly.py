"""
Script to run on the explorer cluster using MCMC to identify best trees for each of the datasets.
"""
import os
import argparse
import mcmc
import assembly_tree as at
import networkx as nx
import numpy as np
import json
import time 
import os
import multiprocessing
print("PID:", os.getpid())
print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))
print("Available CPUs:", multiprocessing.cpu_count())
# Parse command line arguments
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
    start = time.time()
    args = parse_args()
    self_assembly = np.nan
    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Graph name
    graph_name = os.path.splitext(os.path.basename(args.graph_file))[0]

    # Read in target graph
    target = nx.read_edgelist(args.graph_file, nodetype=int)
    # Read in X matrix
    X = np.loadtxt(args.X_file, delimiter=' ')
    # Find O matrix
    #O = at.extract_O(target, X)
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
    # Get whether multiedge
    if args.multiedge == 0:
        multiedge = False
    else:
        multiedge = True
    # Initialize first assembly tree
    print("Create assembly tree object in run_mcmc.py")
    #initial_tree = mcmc.AssemblyTree(target, X, O, capacity, multiedge=multiedge)
    print("Initial assembly tree successfully created in run_mcmc.py")
    if target.number_of_nodes() <= 2:
        # If the graph has 2 or fewer nodes, we can directly return the initial tree
        self_assembly = 1
        output_file = os.path.join(args.output, f"{graph_name}_SA.txt")
        np.savetxt(output_file,np.array([self_assembly]))
        return
    # Run MCMC to find best assembly tree
    print('Run_mcmc before Design MCMC object creation')
    initial_graph = nx.Graph()
    initial_graph.nodes = target.nodes()
    p, samples, idx, success = at.prob_dist(X,O,capacity,T = 2*target.number_of_edges(),initial_graph = initial_graph,rewire_est=True,max_edges=True)
    if len(p) == 1:
        self_assembly = 1
    else:
        self_assembly = 0
    output_file = os.path.join(args.output, f"{graph_name}_SA.txt")
    np.savetxt(output_file,np.array([self_assembly]))
if __name__ == "__main__":
    print('Entered run_mcmc.py')
    main()
    print('finished run_mcmc.py')

# This script is designed to be run on a cluster with the necessary dependencies installed.
# It uses argparse to handle command line arguments for flexibility in specifying input files and parameters.
# Ensure you have the required libraries installed in your environment before running this script.
    

