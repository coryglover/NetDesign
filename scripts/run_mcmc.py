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
    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Graph name
    graph_name = os.path.splitext(os.path.basename(args.graph_file))[0]

    # Read in target graph
    target = nx.read_edgelist(args.graph_file, nodetype=int)
    # Read in X matrix
    X = np.loadtxt(args.X_file, delimiter=' ')
    if X.ndim == 1:
        X = X.reshape(len(X),1)
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
    # CHeck whether current last tree exists
    if os.path.exists(os.path.join(args.output, f"{graph_name}_tree.json")):
        print(f"Last tree already exists in {args.output}")
        # Initialize assembly tree from last tree
        # Load json
        with open(os.path.join(args.output, f"{graph_name}_tree.json"), 'r') as f:
            last_tree = json.load(f)
        if type(last_tree) == list:
            last_tree = last_tree[0]
        initial_tree = mcmc.AssemblyTree(target, X, O, capacity, multiedge=multiedge)
        initial_tree = at.load_tree(last_tree, initial_tree)
    else:    
        print("Create assembly tree object in run_mcmc.py if does not exist")
        initial_tree = mcmc.AssemblyTree(target, X, O, capacity, multiedge=multiedge)
    print("Initial assembly tree successfully created in run_mcmc.py")
    if target.number_of_nodes() <= 2 or max(initial_tree.Tree.get_node(0).data.p) == 1.0:
        # If the graph has 2 or fewer nodes, we can directly return the initial tree
        print("Saving trivial example in run_mcmc.py")
        best_trees_dicts = [initial_tree.Tree.to_dict(with_data=True)]
        for i, tree in enumerate(best_trees_dicts):
            best_trees_dicts[i] = mcmc.expand_tree(tree)
            best_trees_dicts[i]['success'] = 1

        output_tree_file = os.path.join(args.output, f"{graph_name}_tree.json")
        output_tree_stats_file = os.path.join(args.output, f"{graph_name}_tree_stats.txt")
        with open(output_tree_file, 'w') as f:
            json.dump(best_trees_dicts, f, indent=4)
        stats = np.zeros((1,5))
        stats[:,0] = 0
        stats[:,1] = 1
        stats[:,2] = 0
        stats[:,3] = 1
        stats[:,4] = 1
        np.savetxt(output_tree_stats_file, stats, delimiter=',', comments='')
        print("Saved trivial example in run_mcmc.py")
        return
    # Run MCMC to find best assembly tree
    print('Run_mcmc before Design MCMC object creation')
    # Check whether tree can be designed
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from(np.arange(X.shape[0]))
    _, opt_edges = at.find_optimal_edge_count(X, O, capacity, initial_graph=None,solution=False,disp=False,ret_edges=True)
    max_edges = len(opt_edges)

    if max_edges != target.number_of_edges():
        print("Saving impossible example in run_mcmc.py")
        best_trees_dicts = [initial_tree.Tree.to_dict(with_data=True)]
        for i, tree in enumerate(best_trees_dicts):
            best_trees_dicts[i] = mcmc.expand_tree(tree)
            best_trees_dicts[i]['success'] = 1

        output_tree_file = os.path.join(args.output, f"{graph_name}_tree.json")
        output_tree_stats_file = os.path.join(args.output, f"{graph_name}_tree_stats.txt")
        with open(output_tree_file, 'w') as f:
            json.dump(best_trees_dicts, f, indent=4)
        num = 0
        p = 0,
        depth = initial_tree.Tree.depth(),
        num_leaves = len(initial_tree.Tree.leaves()),
        num_nodes = len(initial_tree.Tree.all_nodes())
        #stats = np.array([num, p, depth, num_leaves, num_nodes])
        stats = np.zeros((1,5))
        stats[0,0] = num
        stats[0,1] = 0
        stats[0,2] = 0
        stats[0,3] = 1
        stats[0,4] = 1
        np.savetxt(output_tree_stats_file, stats, delimiter=',', comments='')
        print("Saved impossible example in run_mcmc.py")
        return
    mcmc_obj = mcmc.DesignMCMC(initial_tree)
    print('Run_mcmc after DesignMCMC object creation')
    ratio = args.num_samples // 10
    time_int = args.num_samples // ratio
    #Tis = np.linspace(10,25,args.num_samples)[::-1]
    Tis = np.ones(args.num_samples)
    for i in range(ratio):
        print('Run mcmc before running object')
        mcmc_obj.run_mcmc(time_int,Tis[time_int*i:time_int*(i+1)],dist=[.25,.25,0,0,.25,.25])
        print(f'Run mcmc after running object {i}')
        
        # if time.time() - start > 900:
    

    # Save the results
    # Get the best performing trees
        if not args.MAPonly:
            one_best_sample_idx = np.argmax(mcmc_obj.dist)
            best_samples_idx = np.where(mcmc_obj.dist == mcmc_obj.dist[one_best_sample_idx])[0]
            best_samples = [mcmc_obj.samples[i] for i in best_samples_idx]
        else:
            best_samples = mcmc_obj.best_Ts
            unique_trees = [samples.Tree.to_dict(with_data=True) for samples in best_samples]
            for i,tree in enumerate(unique_trees):
                unique_trees[i] = mcmc.expand_tree(tree)
                unique_trees[i]["success"] = 1 if best_samples[i].success else 0

# Get depths of best performing trees
        # Get depths of best performing trees
        depths = [samples.Tree.depth() for samples in best_samples]
        num_leaves = [len(samples.Tree.leaves()) for samples in best_samples]
        num_nodes = [len(samples.Tree.all_nodes()) for samples in best_samples]
        # Get minimal depth
        min_depth = min(depths)

        if not args.MAPonly:
            best_trees_dicts = [samples.Tree.to_dict(with_data=True) for samples in best_samples]
            # Expand trees
            unique_trees = []
            unique = True
            for i, tree in enumerate(best_trees_dicts):
                unique = True
                best_trees_dicts[i] = mcmc.expand_tree(tree)
                best_trees_dicts[i]['success'] = 1 if best_samples[i].success else 0
                for h in unique_trees:
                    if h == best_trees_dicts[i]:
                        unique = False
                        continue
                if unique:
                    unique_trees.append(best_trees_dicts[i])
# Get trees with minimal depth
#best_trees = [samples for samples in best_samples if samples.Tree.depth() == min_depth]
# Convert to dictionaries
#         best_trees_dicts = [samples.Tree.to_dict(with_data=True) for samples in best_samples]
# # Expand trees
#         unique_trees = []
#         unique = True
#         for i, tree in enumerate(best_trees_dicts):
#             unique = True
#             best_trees_dicts[i] = mcmc.expand_tree(tree)
#             best_trees_dicts[i]['success'] = 1 if best_samples[i].success else 0
#             for h in unique_trees:
#                 if h == best_trees_dicts[i]:
#                     unique = False
#                     continue
#             if unique:
#                  unique_trees.append(best_trees_dicts[i])

# Save the best trees to output directory
        # Save the best trees to output directory
        output_tree_file = os.path.join(args.output, f"{graph_name}_tree.json")
        output_tree_stats_file = os.path.join(args.output, f"{graph_name}_tree_stats.txt")
        
        with open(output_tree_file, 'w') as f:
            json.dump(unique_trees, f, indent=4)
        # Save the statistics of the best trees
        
        if not args.MAPonly:
            stats = np.zeros((1,3))
            stats[:,0] = np.exp(mcmc_obj.dist[one_best_sample_idx])
            stats[:,1] = min_depth
            stats[:,2] = time.time() - start
        else:
            stats = np.zeros((len(unique_trees),5))
            stats[:,0] = np.arange(len(unique_trees))
            stats[:,1] = np.full(len(unique_trees),np.exp(mcmc_obj.best_logp))
            stats[:,2] = depths
            stats[:,3] = num_leaves
            stats[:,4] = num_nodes

            
        np.savetxt(output_tree_stats_file, stats, delimiter=',', comments='')
        if np.exp(mcmc_obj.best_logp) == 1:
            print('Found perfect tree, stopping MCMC')
            break

    if not args.MAPonly:
        one_best_sample_idx = np.argmax(mcmc_obj.dist)
        best_samples_idx = np.where(mcmc_obj.dist == mcmc_obj.dist[one_best_sample_idx])[0]
        best_samples = [mcmc_obj.samples[i] for i in best_samples_idx]
    else:
        best_samples = mcmc_obj.best_Ts
        unique_trees = [samples.Tree.to_dict(with_data=True) for samples in best_samples]
        for i,tree in enumerate(unique_trees):
            unique_trees[i] = mcmc.expand_tree(tree)
            unique_trees[i]["success"] = 1 if best_samples[i].success else 0

    # Get depths of best performing trees
    depths = [samples.Tree.depth() for samples in best_samples]
    num_leaves = [len(samples.Tree.leaves()) for samples in best_samples]
    num_nodes = [len(samples.Tree.all_nodes()) for samples in best_samples]
    # Get minimal depth
    min_depth = min(depths)
    # Get trees with minimal depth
    #best_trees = [samples for samples in best_samples if samples.Tree.depth() == min_depth]
    # Convert to dictionaries
    
    if not args.MAPonly:
        best_trees_dicts = [samples.Tree.to_dict(with_data=True) for samples in best_samples]
        # Expand trees
        unique_trees = []
        unique = True
        for i, tree in enumerate(best_trees_dicts):
            unique = True
            best_trees_dicts[i] = mcmc.expand_tree(tree)
            best_trees_dicts[i]['success'] = 1 if best_samples[i].success else 0
            for h in unique_trees:
                if h == best_trees_dicts[i]:
                    unique = False
                    continue
            if unique:
                unique_trees.append(best_trees_dicts[i])

    # Save the best trees to output directory
    output_tree_file = os.path.join(args.output, f"{graph_name}_tree.json")
    output_tree_stats_file = os.path.join(args.output, f"{graph_name}_tree_stats.txt")
    
    with open(output_tree_file, 'w') as f:
        json.dump(unique_trees, f, indent=4)
    # Save the statistics of the best trees
    
    if not args.MAPonly:
        stats = np.zeros((1,3))
        stats[:,0] = np.exp(mcmc_obj.dist[one_best_sample_idx])
        stats[:,1] = min_depth
        stats[:,2] = time.time() - start
    else:
        stats = np.zeros((len(unique_trees),5))
        stats[:,0] = np.arange(len(unique_trees))
        stats[:,1] = np.full(len(unique_trees),np.exp(mcmc_obj.best_logp))
        stats[:,2] = depths
        stats[:,3] = num_leaves
        stats[:,4] = num_nodes


    np.savetxt(output_tree_stats_file, stats, delimiter=',', comments='')

if __name__ == "__main__":
    print('Entered run_mcmc.py')
    main()
    print('finished run_mcmc.py')

# This script is designed to be run on a cluster with the necessary dependencies installed.
# It uses argparse to handle command line arguments for flexibility in specifying input files and parameters.
# Ensure you have the required libraries installed in your environment before running this script.
    

