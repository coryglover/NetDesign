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

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run MCMC to identify best trees for datasets.")
    parser.add_argument("--graph_file", type=str, required=True, help='Path to specific network')
    parser.add_argument("--X_file", type=str, required=True, help="Path to the X matrix file.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to draw in MCMC.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results.")
    return parser.parse_args()

def main():
    args = parse_args()
    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read in target graph
    target = nx.read_edgelist(args.graph_file, nodetype=int)
    # Read in X matrix
    X = np.loadtxt(args.X_file, delimiter=' ')
    # Find O matrix
    O = at.extract_O(target, X)
    # Get capacity vector
    capacity = at.extract_deg_cap(target, X).reshape(-1)
    # Initialize first assembly tree
    initial_tree = mcmc.AssemblyTree(target, X, O, capacity)

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
    best_trees = [samples for samples in best_samples if samples.Tree.depth() == min_depth]
    # Convert to dictionaries
    best_trees_dicts = [samples.Tree.to_dict(with_data=True) for samples in best_trees]
    # Expand trees
    for tree in best_trees_dicts:
        mcmc.expand_tree(tree)

    # Save the best trees to output directory
    output_tree_file = os.path.join(args.output, "best_trees.json")
    output_tree_stats_file = os.path.join(args.output, "best_tree_stats.txt")

    with open(output_tree_file, 'w') as f:
        json.dump(best_trees_dicts, f, indent=4)
    # Save the statistics of the best trees
    stats = np.zeros((1,2))
    stats[:,0] = mcmc_obj.dist[one_best_sample_idx]
    stats[:,1] = min_depth
    np.savetxt(output_tree_stats_file, stats, delimiter=',', header='p,depth', comments='')
    
if __name__ == "__main__":
    main()

# This script is designed to be run on a cluster with the necessary dependencies installed.
# It uses argparse to handle command line arguments for flexibility in specifying input files and parameters.
# Ensure you have the required libraries installed in your environment before running this script.
    