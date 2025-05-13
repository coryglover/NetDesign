import assembly_tree as at
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import random
import sympy
from scipy.special import stirling2
import networkx as nx
import treelib

def get_integer_partition(n,m=None):
    """Return a uniformly random integer partition of n."""
    partitions_dicts = list(sympy.utilities.iterables.partitions(n,m=m))
    
    # Convert from dicts (e.g. {3:1, 1:2}) to sorted list [3,1,1]
    partitions = []
    for p in partitions_dicts:
        part = []
        for k, v in p.items():
            part.extend([k] * v)
        partitions.append(sorted(part, reverse=True))
    if m != None:
        return random.choice([p for p in partitions if len(p)==m])
    else:
        return random.choice(partitions)

class AssemblyTree:
    """
    AssemblyTree class for generating assembly trees.
    This class implements the assembly tree structure and methods for splitting, merging, and redistributing nodes.
    
    Attributes
    ----------
    count : int -- The number of nodes in the tree.
    parent : TreeNode2 -- The parent node of the current node.
    children : list -- List of child nodes.
    """
    def __init__(self, target, X, O, capacity):
        """
        Initialize the AssemblyTree class with the number of nodes.
        
        Parameters
        ----------
        count : int -- The number of nodes in the tree.
        """
        self.nodes = list(target.nodes())
        self.X = X
        self.O = O
        self.capacity = capacity
        self.Tree = treelib.Tree()
        self.target = target
        self.target_A = nx.adjacency_matrix(self.target).todense()
        # Initial graph 
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.Tree.create_node(data=AssemblyNode(self.nodes,self.X,self.O,self.capacity,subgraph=[]),identifier=self.Tree.size())
        self.update_prob(0)

    def update_tree(self,dist=None):
        """
        Perform some update step on nodes

        Parameters
        ----------
        dist (np.ndarray) -- Probability distribution for updating operations.
        """
        # Get list of old leaves 
        old_leaves = self.Tree.leaves()
        if dist is None:
            operation = np.random.choice(['split','merge','redistribute','complete_branch'],p=[0.25,0.25,0.25,0.25])
        else:
            operation = np.random.choice(['split','merge','redistribute','complete_branch'],p=dist)
        # Perform opertation
        if operation == 'split':
            # Get node ID
            node_id = random.choice(self.Tree.leaves()).identifier
            # Split node
            self.split(node_id)
        elif operation == 'merge':
            # Get node ID
            leaves = self.Tree.leaves()
            # Choose from parents of leaves
            node_id = random.choice([self.Tree.parent(leaf.identifier).identifier for leaf in leaves])
            # Merge node
            self.merge(node_id)
        elif operation == 'redistribute':
            # Get node ID
            node_id = random.choice(self.Tree.leaves()).identifier
            # Redistribute node
            self.redistribute(node_id)
        elif operation == 'complete_branch':
            # Get node ID
            node_id = random.choice(self.Tree.leaves()).identifier
            # Complete branch
            self.complete_branch(node_id)
        # Update probability of node assembling into a subgraph of G
        # Get children of node

        # Update relevant leaf nodes
        new_leaves = self.Tree.leaves()
        # Get difference between old and new leaves
        nodes_to_update = [leaf for leaf in new_leaves if leaf not in old_leaves]
        # Update probability of all new leaves
        while len(nodes_to_update) > 0:
            leaf = nodes_to_update.pop()
            self.update_prob(leaf.identifier)
            if self.Tree.parent(leaf.identifier) is not None:
                # Get parent node
                parent = self.Tree.parent(leaf.identifier)
                nodes_to_update.append(parent)

    def split(self,node_id):
        """
        Split the assembly tree at the given node ID.
        
        Parameters
        ----------
        node_id : int -- The ID of the node to split.
        """
        count = self.Tree.get_node(node_id).data.count
        n_children = random.randint(2,count-1)
        cont = True
        while(cont):
            counts_children = get_integer_partition(count,n_children)
            if len(counts_children)<count and len(counts_children)>1:
                cont=False
        # Get graph node in tree node
        node = self.Tree.get_node(node_id)
        graph_node_list = node.data.nodes
        # Randomly order nodes
        random.shuffle(graph_node_list)
        # Create children nodes
        for i in range(1,len(counts_children)+1):
            # Get nodes for child
            graph_nodes = graph_node_list[sum(counts_children[:(i-1)]):sum(counts_children[:i])]            # Create child node
            # Create graph
            G = nx.Graph()
            G.add_nodes_from(graph_nodes)
            child = AssemblyNode(graph_nodes, self.X, self.O, self.capacity, subgraph = [])
            # Add child to tree
            self.Tree.create_node(data=child, parent=node_id, identifier=self.Tree.size())
        
    def merge(self,node_id):
        """
        Merge the assembly tree at the given node ID.
        
        Parameters
        ----------
        node_id : int -- The ID of the node to merge.
        """
        # Remove children from tree
        for child in self.Tree.children(node_id):
            self.Tree.remove_node(child.identifier)
    
    def redistribute(self,node_id):
        """
        Redistribute the size of children of a given node.
        
        Parameters
        ----------
        node_id : int -- The ID of the node to redistribute.
        """
        # Merge children
        self.merge(node_id)
        # Split into new children
        self.split(node_id)

    def complete_branch(self,node_id):
        """
        Expand branch until all treenodes have 2 or less graph nodes
        
        Parameters
        ----------
        node_id : int -- The ID of the node to complete.
        """
        # Get graph node in tree node
        if len(self.Tree.get_node(node_id).data.nodes) <= 2:
            pass

        nodes_to_split = [node_id]
        # Recurivsely split children until all children have no more than 2 nodes in their data
        while len(nodes_to_split) > 0:
            cur_node = nodes_to_split.pop()
            self.split(cur_node)
            # Get children
            children = self.Tree.get_node(cur_node).successors(self.Tree.identifier)
            for child in children:
                if len(self.Tree.get_node(child).data.nodes) > 2:
                    nodes_to_split.append(child)

    def update_prob(self,node_id):
        """
        Update the probability of a given node assembling into a subgraph of G
        and all of its parents.
        
        Parameters
        ----------
        node_id : int -- The ID of the node to update.
        """
        # Get graph node in tree node
        node = self.Tree.get_node(node_id)
        # Reset probabilities
        node.data.logP = []
        node.data.subgraph = []
        # If leaf node
        if len(node.data.subgraph) == 0:
            # Get nodes of subgraph
            sub_nodes = node.data.nodes
            # Get probability of node assembling into a subgraph of G
            p, samples, idx = at.prob_dist(self.X[sub_nodes,:], self.O, self.capacity, initial_graph=None,max_edges=True,labeled=True)
            logp = np.log(p / p.sum())
            
            # Get true adjacency matrix
            true_A = self.target_A[:,sub_nodes]
            true_A = true_A[sub_nodes,:]
            # Find subgraphs of true graph
            for i in idx:
                # Get adjacency matrix
                cur_A = nx.adjacency_matrix(samples[i]).todense()
                if np.allclose(true_A,cur_A):
                    node.data.logP.append(logp[i])
                    node.data.subgraph.append(samples[i])

        # Loop through all existing subgraphs as starting points
        else:
            for cur_subgraph in node.data.subgraph:
                # Get nodes of subgraph
                sub_nodes = node.data.nodes
                # Get probability of node assembling into a subgraph of G
                p, samples, idx = at.prob_dist(self.X[sub_nodes,:], self.O, self.capacity, initial_graph=cur_subgraph,max_edges=True,labeled=True)
                logp = np.log(p / p.sum())
                # Get true adjacency matrix
                true_A = self.target_A[sub_nodes,:][:,sub_nodes]
                # Find subgraphs of true graph
                for i in idx:
                    # Get adjacency matrix
                    cur_A = nx.adjacency_matrix(samples[i]).todense()
                    if np.allclose(true_A,cur_A):
                        node.data.logP.append(logp[i])
                        node.data.subgraph.append(samples[i])

        
        

class AssemblyNode():
    """
    AssemblyNode class for representing nodes in the assembly tree.
    This class implements the node structure and methods for splitting, merging, and redistributing nodes.
    
    Attributes
    ----------
    nodes : list -- List of graph nodes in the assembly node.
    subgraph : nx.Graph -- The subgraph formed by the assembly node.
    P : float -- Probability of the node assembling into a subgraph of G.
    """
    def __init__(self, nodes, X, O, capacity, subgraph=[], logP=[]):
        """
        Initialize the AssemblyNode class with the nodes, subgraph, and probability.
        
        Parameters
        ----------
        nodes : list -- List of graph nodes in the assembly node.
        subgraph : nx.Graph -- The subgraph formed by the assembly node.
        P : float -- Probability of the node assembling into a subgraph of G.
        """
        self.nodes = nodes
        self.X = X
        self.O = O
        self.capacity = capacity
        self.subgraph = subgraph
        self.logP = logP
        self.count = len(nodes)

class TreeNode2:
    def __init__(self, count, parent=None):
        self.count = count
        self.parent = parent
        self.children = []
        self.P = np.nan
    
    def add_children(self):
        n_children = random.randint(2,self.count-1)
        cont = True
        while(cont):
            counts_children = get_integer_partition(self.count,n_children)
            if len(counts_children)<self.count and len(counts_children)>1:
                cont=False

        for i in range(len(counts_children)):
            child = TreeNode2(counts_children[i],parent=self)
            self.children.append(child)
    
    def remove_children(self):
        self.children = []

    def split(self,leavesl,leavess,leaves2,treenodes):
        self.add_children()
        for child in self.children:
            treenodes.add(child)
            if child.count <=2:
                leavess.add(child)
            else:
                leavesl.add(child)
        leavesl.remove(self)
        leaves2.add(self)
        if self.parent in leaves2:
            leaves2.remove(self.parent)

    def merge(self,leavesl,leavess,leaves2,treenodes):
        for child in self.children:
            treenodes.remove(child)
            if child.count <=2:
                leavess.remove(child)
            else:
                leavesl.remove(child)
        self.remove_children()
        leavesl.add(self)
        leaves2.remove(self)
        check = True
        if self.parent != None:
            for child in self.parent.children:
                if child not in leavesl and child not in leavess:
                    check = False
            if check:
                leaves2.add(self.parent)

    def complete_branch(self,leavesl,leavess,leaves2,treenodes):
        nodeleavess = set()
        nodeleavesl = set()
        nodeleavesl.add(self)
        leavesl.remove(self)
        while(len(nodeleavesl)>0):
            node2 = random.choice(list(nodeleavesl))
            node2.split(nodeleavesl,nodeleavess,leaves2,treenodes)
        for leaf in nodeleavess:
            leavess.add(leaf)

    def redistribute(self,leavesl,leavess,leaves2,treenodes):
        n_children = len(self.children)
        for child in self.children:
            if child.count > 2:
                leavesl.remove(child)
            else:
                leavess.remove(child)
        counts_children = get_integer_partition(self.count,n_children)

        for i,child in enumerate(self.children):
            child.count = counts_children[i]
            if child.count > 2:
                leavesl.add(child)
            else:
                leavess.add(child)

def encode_tree(node,heights):
    # Base case: if the node is a leaf, return just the count
    if not hasattr(node, "children") or not node.children:
        return [node.count]
    
    # Otherwise, encode the subtree for each child
    encoded_children = [encode_tree(child,heights) for child in sorted(node.children,key = lambda v: (-heights[v],-v.count))]
    
    return [node.count] + [encoded_children]

def find_height(node, level=0):
    if not node.children:
        return level
    return max(find_height(child, level + 1) for child in node.children)


root = TreeNode2(9)
leavesl = set()
leavess = set()
leaves2 = set()
leavesl.add(root)
treenodes = set()
treenodes.add(root)
T = 1000

for i in range(T):
    if len(leaves2) == 0 and len(leavesl) != 0:
        j = random.choice([1,3])
    elif len(leaves2) != 0 and len(leavesl) == 0:
        j = random.choice([2,4])
    elif len(leaves2) != 0 and len(leavesl) != 0:
        j = random.choice([1,2,3,4])
    else:
        print("HERE")
    if j==1:
        node = random.choice(list(leavesl))
        node.split(leavesl,leavess,leaves2,treenodes)
    if j==2:
        node = random.choice(list(leaves2))
        node.merge(leavesl,leavess,leaves2,treenodes)
    if j==3:
        node = random.choice(list(leavesl))
        node.complete_branch(leavesl,leavess,leaves2,treenodes)
    if j==4:
        node = random.choice(list(leaves2))
        node.redistribute(leavesl,leavess,leaves2,treenodes)

heights = {v:find_height(v,0) for v in treenodes}
tree = encode_tree(root,heights)
# print(find_height(root))

class DesignMCMC:
    """
    DesignMCMC class for performing MCMC sampling on assembly trees.
    This class implements the Metropolis-Hastings algorithm to sample from the posterior distribution of assembly trees given a target graph and binding matrix.


    Attributes
    ----------
    T : numpy.ndarray -- The assembly tree.
    G : nx.Graph -- The target graph.
    X : numpy.ndarray -- Node labels.
    O : numpy.ndarray -- Binding matrix.
    """

    def __init__(self, T, G, X, O):
        """
        Initialize the DesignMCMC class with the assembly tree, target graph, node labels, and binding matrix.
        """
        self.T = T
        self.G = G
        self.X = X
        self.O = O
        pass

    def run_mcmc(self, num_samples):
        """
        Run the MCMC sampling algorithm.
        
        Parameters
        ----------
        num_samples : int -- Number of samples to generate.

        Returns
        -------
        samples : list -- List of sampled assembly trees.
        """
        samples = []
        for i in range(num_samples):
            # Propose a new tree
            new_tree = self.propose_new_tree()
            # Compute acceptance probability
            acceptance_prob = self.compute_acceptance_prob(new_tree)
            # Accept or reject the new tree
            if np.random.rand() < acceptance_prob:
                self.T = new_tree
            samples.append(self.T)
        return samples
    
    

    def posterior(T,G,X,O):
        """
        Compute the posterior distribution of the parameters given the data.
        
        Parameters
        ----------
        T : numpy.ndarray -- The assembly tree.
        G : nx.Graph -- The target graph.
        X : numpy.ndarray -- Node labels.
        O : numpy.ndarray -- Binding matrix.

        Returns
        -------
        posterior_val : float -- The posterior distribution value.
        """
        pass

    def recursive_summation(depth, current_Q):
        if depth == 2:
            return stirling.s2(current_Q, 2)
        summation = 0
        for an in range(depth, current_Q - 1):
            s = stirling.s2(current_Q, an)
            summation += s + recursive_summation(depth - 1, an)
        return summation

    def tree_prior(T,X):
        """
        Compute the prior distribution of the assembly tree.
        
        Parameters
        ----------
        T : list -- List of tree nodes in assembly tree.
        X : numpy.ndarray -- Node labels.

        Returns
        -------
        prior_val : float -- The prior distribution value.
        """
        # Get depth of tree
        D = np.max([find_height(node) for node in T])
        # Get number of partitions
        Q = num_leaves(T)

        # Compute Bell Number on N nodes
        N = X.shape[0]
        bell_number = 0
        for k in range(N + 1):
            bell_number += stirling2(N, k)
        
        # Calculate internal count of number of possible splits with depth D
        num_trees = recursive_summation(D, Q)

        return -np.log(bell_number) - np.log(Q) - np.log(num_trees)

    def prob_tree_node(G,node,X,O,capacity,initial_graph=None,**kwargs):
        """
        Compute the probability of a given tree node assembling into a subgraph of G.
        
        Parameters
        ----------
        G : nx.Graph -- The target graph.
        node : TreeNode2 -- The tree node.
        X : numpy.ndarray -- Node labels.
        O : numpy.ndarray -- Binding matrix.
        initial_graph : nx.Graph, optional -- The initial graph (default is None).
        
        Returns
        -------
        prob : float -- The probability of the tree node assembling into a subgraph of G.
        """
        # Simulate assembly of node for 1000 time steps
        p, samples, idx = at.prob_dist(X,O,capacity,initial_graph=initial_graph,**kwargs)
        # Check whether sample is subgraph of G
        for s in samples[idx]:
            if nx.is_isomorphic(s,nx.subgraph(G,s.nodes())):
                return p

    def likelihood(G,T,X,O):
        """
        Compute the likelihood of the data given the parameters.
        
        Parameters
        ----------
        G : nx.Graph -- The target graph.
        T : numpy.ndarray -- The assembly tree.
        X : numpy.ndarray -- Node labels.
        O : numpy.ndarray -- Binding matrix.

        Returns
        -------
        likelihood_val : float -- The likelihood value.
        """
        # Create dictionary of tree assemblies
        pass

