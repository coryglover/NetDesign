import pandas as pd
import numpy as np
import networkx as nx
import json
import treelib
import os
import argparse

# Read in arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Create a DataFrame from a JSON file.")
    parser.add_argument("--dir", type=str, help="Path to directory containing data.")
    return parser.parse_args()

def read_json_tree(json_tree):
    """
    Read JSON tree into treelib object

    Parameters
    ----------
    json_tree : str
        Path to the JSON file containing the tree structure
    
    Returns
    -------
    tree : treelib.Tree
        A treelib tree object representing the structure
    """
    def add_children(tree,parent,children):
        """
        Recursively add children to the tree
        
        Parameters
        ----------
        tree : treelib.Tree
            The tree to which children will be added
        parent : str
            The identifier of the parent node
        children : list
            List of child nodes to be added under the parent
        """
        for child in children:
            node_name = list(child.keys())[0]
            tree.create_node(tag=node_name, identifier=node_name, data=child[node_name]['data'], parent=parent)
            if 'children' in child[node_name]:
                add_children(tree, node_name, child[node_name]['children'])

    with open(json_tree, 'r') as f:
        data = json.load(f)
    tree_list = []
    for tree_data in data:
        root = list(tree_data.keys())[0]
        tree_data[root]['data']
        tree = treelib.Tree()
        tree.create_node(tag=root, identifier=root, data=tree_data[root]['data'])
        if 'children' in tree_data[root]:
            children = tree_data[root]['children']
            add_children(tree, root, children)
        tree_list.append(tree)
    return tree_list

args = parse_args()
print(args)
with open('/scratch/glover.co/NetDesign/data/properties.json','r') as f:
    properties = json.load(f)
properties = properties['proteins']
print(properties)
# Initialize dataframe
df = pd.DataFrame(columns=['type',
                           'basename',
                           'N',
                           'avg_k',
                           'cyclicity',
                           'node_symmetry',
                           'graph_density',
                           'diversity',
                           'degree_heterogeneity',
                           'clustering',
                           'Oness',
                           'N_types',
                           'tree_num',
                           'depth',
                           'leaves'])

# Get subdirectory names
subdirs = os.listdir(args.dir)
# subdirs = [d for d in subdirs if os.path.isdir(os.path.join(args.dir, d))]
print('here')
# Get all names from the directory
for subdir in subdirs:
    print(subdir)
    edgefiles = os.listdir(args.dir + subdir + '/edgefiles/')
    Xfiles = os.listdir(args.dir + subdir + '/Xfiles/')
    treefiles = os.listdir(args.dir + subdir + '/treefiles/')
    basenames = [f[:-5] for f in edgefiles]

    for i, basename in enumerate(basenames):
        # Read edge file
        print(basename)
        with open(args.dir + subdir + '/edgefiles/' + basename + '.edge', 'r') as f:
            g = nx.read_edgelist(f)
        # Read X file
        with open(args.dir + subdir + '/Xfiles/X_' + basename + '.txt', 'r') as f:
            X = np.loadtxt(f)
        # Read tree file
        try:
            f = args.dir + subdir + '/treefiles/' + basename + '_tree.json'
            tree = read_json_tree(f)
        except:
            continue
        # Calculate properties
        property_dict = properties[subdir][basename]
        for j,t in enumerate(tree):
            data = []
            data.append(subdir)
            data.append(basename)
            data.append(property_dict['N'])
            data.append(property_dict['kavg'])
            data.append(property_dict['cyclicity'])
            data.append(property_dict['node_symmetry'])
            data.append(property_dict['graph_density'])
            data.append(property_dict['node_heterogeneity'])
            data.append(property_dict['degree_heterogeneity'])
            data.append(property_dict['clustering'])
            data.append(property_dict['Oness'])
            data.append(X.shape[1])
            data.append(j)
            data.append(t.depth())
            data.append(len(t.leaves()))
            df.loc[len(df)] = data
# Save DataFrame to CSV
df.to_csv(args.dir + 'protein_dataframe.csv', index=False)

#if __name__ == "__main__":
#    main()



