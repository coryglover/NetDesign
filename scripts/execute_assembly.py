import assembly_tree as at
import numpy as np
import networkx as nx

# Read in example protein
protein = nx.read_edgelist('../data/protein_complex/proteins/human/edgefiles/CPX-1919.edge',nodetype=int)
X = np.loadtxt('../data/protein_complex/proteins/human/Xfiles/X_CPX-1919.txt')

p, samples,idx = at.prob_dist(X,O,O.sum(axis=1),max_iters=10000,verbose=True)

np.savetxt('/scratch/glover.co/NetDesign/data/fig1/c1q/p.txt')
for i in range(len(samples)):
    nx.write_edgelist(samples[i],f'/scratch/glover.co/NetDesign/data/fig1/c1q/samples/sample_{i}.edge',data=False)


