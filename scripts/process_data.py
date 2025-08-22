import os
import numpy as np 
import networkx as nx 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process data for graph analysis.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input files.')
    parser.add_argument('--output', type=str, required=True, help='File to save graph names.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Get subdirectories
    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    # Initialize list
    graph_names = []

    # Process each subdirectory
    for subdir in subdirs:
        # Get subdir
        subdir_path = os.path.join(args.input_dir, subdir)
        
        # Get edgefiles
        edgefiles_path = os.path.join(subdir_path, 'edgefiles')
        if not os.path.exists(edgefiles_path):
            continue
        edgefiles = [f for f in os.listdir(edgefiles_path) if f.endswith('.edge')]
        # Process each edgefile
        for edgefile in edgefiles:
            edgefile_path = os.path.join(edgefiles_path, edgefile)
            if not os.path.exists(edgefile_path):
                continue

            graph_name = os.path.join(subdir, 'edgefiles', edgefile)
            # Get label matrix
            X_path = os.path.join(subdir_path, 'Xfiles', f'X_{edgefile[:-5]}.txt')
            X = np.loadtxt(X_path, dtype=int)

            G = nx.MultiGraph()
            G.add_nodes_from(np.arange(X.shape[0]))

            # Add edges from the edge file
            with open(edgefile_path, 'r') as f:
                for line in f:
                    u, v = map(int, line.strip().split())
                    G.add_edge(u, v)
            
            if G.number_of_edges() == 1:
                print(f"Graph {edgefile} has one edge, skipping.")
                continue
            
            # Check for multiedges
            has_multiedges = any(G.number_of_edges(u, v) > 1 for u, v in G.edges())
            if has_multiedges:
                print(f"Graph {edgefile} has multiedges, skipping.")
                continue
            else:
                g = nx.Graph(G)
            
            # Check number of nodes
            if g.number_of_nodes() <= 2:
                print(f"Graph {edgefile} has 2 or fewer nodes, skipping.")
                continue

            if len(list(nx.selfloop_edges(g))) > 0:
                print(f"Graph {edgefile} has self-loops, skipping.")
                continue

            # Get graph name
            graph_names.append(graph_name)
    
    # Save graph names
    with open(args.output, 'w') as f:
        for name in graph_names:
            f.write(f"{name}\n")
    return

if __name__ == "__main__":
    main()

