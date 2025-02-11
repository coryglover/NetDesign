import pandas as pd
import networkx as nx
import argparse

# Accept input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--file',type=str,
                    default='../data/protein_complex/corum_hint_combined.txt',
                    help='File with dataframe')
parser.add_argument('--graph_dir',type=str,
                    default='../data/protein_complex/graphs/',
                    help='Directory for graphs')

args = parser.parse_args()
file = args.file
graph_dir = args.graph_dir

# Read in dataframe
df = pd.read_csv(file,index_col=False)

# Make networks
complex_ids = pd.unique(df['complex_id'])

for i in complex_ids:
    # Get rows
    cur_id = df[df.complex_id == i]
    g = nx.Graph()
    for index, row in cur_id.iterrows():
        g.add_edge(row['Uniprot_A'],row['Uniprot_B'])
    if g.number_of_nodes() <= 1:
        continue
    nx.write_edgelist(g,f'{graph_dir}/g_{i}.txt')