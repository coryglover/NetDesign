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

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run MCMC to identify best trees for datasets.")
    parser.add_argument("--graph_file", type=str, required=True, help='Path to specific network')
    parser.add_argument("--X_file", type=str, required=True, help="Path to the X matrix file.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to draw in MCMC.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results.")
    parser.add_argument("--multiedge", type=int, required=False, default = 0, help="1 if multiedge")
    return parser.parse_args()

def main():
    start = time.time()
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
    # Find O matrix
    O = at.extract_O(target, X)
    # Get capacity vector
    capacity = at.extract_deg_cap(target, X).reshape(-1)
    # Get whether multiedge
    if args.multiedge == 0:
        multiedge = False
    else:
        multiedge = True
    # Initialize first assembly tree
    initial_tree = mcmc.AssemblyTree(target, X, O, capacity, multiedge=multiedge)
    if target.number_of_nodes() <= 2:
        # If the graph has 2 or fewer nodes, we can directly return the initial tree
        best_trees_dicts = [initial_tree.Tree.to_dict(with_data=True)]
        for d in best_trees_dicts:
            mcmc.expand_tree(d)
        output_tree_file = os.path.join(args.output, f"{graph_name}_tree.json")
        output_tree_stats_file = os.path.join(args.output, f"{graph_name}_tree_stats.txt")
        with open(output_tree_file, 'w') as f:
            json.dump(best_trees_dicts, f, indent=4)
        stats = np.zeros((1,2))
        stats[:,0] = 1.0
        stats[:,1] = initial_tree.Tree.depth()
        np.savetxt(output_tree_stats_file, stats, delimiter=',', header='p,depth', comments='')
        return
    # Run MCMC to find best assembly tree
    mcmc_obj = mcmc.DesignMCMC(initial_tree)
    mcmc_obj.run_mcmc(args.num_samples)

    # Save the results
    # Get the best performing trees
    one_best_sample_idx = np.argmax(mcmc_obj.dist)
    best_samples_idx = np.where(mcmc_obj.dist == mcmc_obj.dist[one_best_sample_idx])[0]
    best_samples = [mcmc_obj.samples[i] for i in best_samples_idx]

    # Get depths of best performing trees
    depths = [samples.Tree.depth() for samples in best_samples]
    # Get minimal depth
    min_depth = min(depths)
    # Get trees with minimal depth
    #best_trees = [samples for samples in best_samples if samples.Tree.depth() == min_depth]
    # Convert to dictionaries
    best_trees_dicts = [samples.Tree.to_dict(with_data=True) for samples in best_samples]
    # Expand trees
    for i, tree in enumerate(best_trees_dicts):
        best_trees_dicts[i] = mcmc.expand_tree(tree)

    # Save the best trees to output directory
    output_tree_file = os.path.join(args.output, f"{graph_name}_tree.json")
    output_tree_stats_file = os.path.join(args.output, f"{graph_name}_tree_stats.txt")

    with open(output_tree_file, 'w') as f:
        json.dump(best_trees_dicts, f, indent=4)
    # Save the statistics of the best trees
    stats = np.zeros((1,3))
    stats[:,0] = mcmc_obj.dist[one_best_sample_idx]
    stats[:,1] = min_depth
    stats[:,2] = time.time() - start
    np.savetxt(output_tree_stats_file, stats, delimiter=',', header='p,depth', comments='')
    
if __name__ == "__main__":
    main()

# This script is designed to be run on a cluster with the necessary dependencies installed.
# It uses argparse to handle command line arguments for flexibility in specifying input files and parameters.
# Ensure you have the required libraries installed in your environment before running this script.
    
