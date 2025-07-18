import assembly_tree as at
import numpy as np
import random
import sympy
from scipy.special import stirling2
import networkx as nx
import treelib
from itertools import product
import copy
from tqdm import tqdm

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
    '''
    AssemblyTree class for generating and managing assembly trees.
    This class implements an assembly tree structure with methods for splitting, merging, redistributing, 
    and completing branches of the tree. It also provides functionality for updating the probability of 
    nodes assembling into subgraphs of a target graph.
    nodes : list
        List of nodes in the target graph.
    X : np.ndarray
        Input data matrix.
    O : np.ndarray
        Output data matrix.
    capacity : int
        Capacity constraint for the assembly process.
    Tree : treelib.Tree
        Tree structure representing the assembly tree.
    target : nx.Graph
        Target graph for the assembly process.
    target_A : np.ndarray
        Adjacency matrix of the target graph.
    G : nx.Graph
        Graph representing the current state of the assembly process.
    Methods
    -------
    __init__(target, X, O, capacity)
        Initialize the AssemblyTree with the target graph, input/output matrices, and capacity.
    update_tree(dist=None)
        Perform an update step on the tree using a specified probability distribution for operations.
    split(node_id)
        Split a node in the tree into multiple child nodes.
    merge(node_id)
        Merge all child nodes of a given node into the parent node.
    redistribute(node_id)
    complete_branch(node_id)
        Expand a branch until all tree nodes have at most two graph nodes.
    update_prob(node_id)
        Update the probability of a node assembling into a subgraph of the target graph.
    '''
    def __init__(self, target, X, O, capacity, multiedge=False):
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
        self.multiedge = multiedge
        # Initial graph 
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.Tree.create_node(data=AssemblyNode(self.nodes,self.X,self.O,self.capacity,subgraph=[]),identifier=self.Tree.size())
        self.success = True
        self.update_prob(0)
        

    def update_tree(self,dist=None,max_iters=100):
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
            # Make list of leaves with more than 2 nodes
            leaves = self.Tree.leaves()
            leaves = [leaf for leaf in leaves if (len(leaf.data.nodes) >2 )]
            if len(leaves) == 0:
                # print('No update performed')
                return False
            # Get node ID
            #probs = [len(leaf.data.nodes) for leaf in leaves]
            #probs = np.array(probs) / sum(probs)
            probs = 1/len(leaves) * np.ones(len(leaves))
            idx = np.random.choice(len(leaves),p=probs)
            node_id = leaves[idx].identifier
            # Split node
            self.split(node_id)
        elif operation == 'merge':
            # Find nodes which are parents of only leaves
            if self.Tree.size() == 1:
                # print('No update performed')
                return False
            # Get leaves
            leaves = self.Tree.leaves()
            
            # Choose from parents of leaves
            parents_of_leaves = [self.Tree.parent(leaf.identifier).identifier for leaf in leaves]
            pos_nodes = []
            # Check if all of parents children are leaves
            for p in parents_of_leaves:
                if all(child.is_leaf() for child in self.Tree.children(p)):
                    pos_nodes.append(p)
            # Choose leaf
            if len(pos_nodes) == 0:
                # print('No update performed')
                return False
            node_id = random.choice(pos_nodes)
            # Merge node
            self.merge(node_id)
        elif operation == 'redistribute':
            if self.Tree.size() == 1:
                # print('No update performed')
                return False
            # Make list of leaves with more than 2 nodes
            leaves = self.Tree.leaves()
            # Choose from parents of leaves
            parents_of_leaves = [self.Tree.parent(leaf.identifier).identifier for leaf in leaves]
            pos_nodes = []
            # Check if all of parents children are leaves
            for p in parents_of_leaves:
                if all(child.is_leaf() for child in self.Tree.children(p)):
                    pos_nodes.append(p)
            # Choose leaf
            if len(pos_nodes) == 0:
                # print('No update performed')
                return False
            node_id = random.choice(pos_nodes)
            # Redistribute node
            self.redistribute(node_id)
        elif operation == 'complete_branch':
            # Make list of leaves with more than 2 nodes
            leaves = self.Tree.leaves()
            leaves = [leaf for leaf in leaves if len(leaf.data.nodes) > 2]
            if len(leaves) == 0:
                # print('No update performed')
                return False
            # Get node ID
            #probs = [len(leaf.data.nodes) for leaf in leaves]
            #probs = np.array(probs) / sum(probs)
            probs = 1 / len(leaves) * np.ones(len(leaves))
            idx = np.random.choice(len(leaves),p=probs)
            node_id = leaves[idx].identifier
        
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
            leaf = nodes_to_update.pop(0)
            self.update_prob(leaf.identifier,max_iters=max_iters)
            if self.Tree.parent(leaf.identifier) is not None:
                # Get parent node
                parent = self.Tree.parent(leaf.identifier)
                nodes_to_update.append(parent)
                # Order nodes by depth
                nodes_to_update = sorted(nodes_to_update, key=lambda x: self.Tree.depth(x), reverse=True)

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
            # Get next node label
            all_nodes = self.Tree.all_nodes()
            node_names = [int(n.identifier) for n in all_nodes]
            next_node = np.max(node_names) + 1
            # Add child to tree
            self.Tree.create_node(data=child, parent=node_id, identifier=next_node)
        
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
            cur_node = nodes_to_split.pop(0)
            self.split(cur_node)
            # Get children
            children = self.Tree.get_node(cur_node).successors(self.Tree.identifier)
            for child in children:
                if len(self.Tree.get_node(child).data.nodes) > 2:
                    nodes_to_split.append(child)

    def update_prob(self,node_id,prob_tol=10e-5,max_iters=1000):
        """
        Update the probability of a given node assembling into a subgraph of G
        and all of its parents.
        
        Parameters
        ----------
        node_id : int -- The ID of the node to update.
        """
        # Get graph node in tree node
        node = self.Tree.get_node(node_id)
        # Get sub_nodes
        sub_nodes = node.data.nodes
        # Reset probabilities
        node.data.p = []
        node.data.subgraph = []
        # Get children nodes
        children = node.successors(self.Tree.identifier)
        # Check whether any children have probability 0 and finish
        total_child_prob = np.prod(np.array([np.sum(x) for c in children for x in self.Tree.get_node(c).data.p]))
        if total_child_prob == 0:
            node.data.p.append(0)
            node.data.subgraph.append(None)
        # If no children, simulate leaf node
        elif node.is_leaf() or self.Tree.depth() == 0:
            # Create initial graph with sub_nodes
            initial_graph = nx.Graph()
            initial_graph.add_nodes_from(sub_nodes)
            # Run simulation
            p, samples, idx, success = at.prob_dist(self.X,self.O,self.capacity,initial_graph=initial_graph,max_edges=True,max_iters=max_iters,rewire_est=True,multiedge=self.multiedge)
            p = p / sum(p)
            for i,subgraph in enumerate(samples):
                # Check whether probability is zero
                if p[i] < prob_tol:
                    node.data.p.append(0)
                    node.data.subgraph.append(None)
                    node.data.success = success
                    if success is False:
                        self.success = False
                    continue
                iso_object = nx.isomorphism.GraphMatcher(self.target,subgraph)
                if iso_object.subgraph_is_isomorphic():
                    node.data.p.append(p[i])
                    node.data.subgraph.append(subgraph)
                    node.data.success = success
                    if success is False:
                        self.success = False
                    break
                else:
                    node.data.p.append(0)
                    node.data.subgraph.append(subgraph)
                    node.data.success = success
                    if success is False:
                        self.success = False
        else:
            # Get all possible subgraph combinations
            subgraphs = [self.Tree.get_node(c).data.subgraph for c in children]
            probs = [self.Tree.get_node(c).data.p for c in children]
            # Get all possible combinations of subgraphs
            subgraph_combinations = list(product(*subgraphs))
            probs_combinations = list(product(*probs))
            # Get probability of all children occuring
            if len(subgraph_combinations) == 1:
                probs = np.array(probs_combinations[0])
            else:
                probs = np.prod(np.array(probs_combinations),axis=1)
        
            # Loop through subgraph combinations
            for i,subgraph in enumerate(subgraph_combinations):
                # Check whether probability is zero
                if probs[i] < prob_tol:
                    node.data.p.append(0)
                    node.data.subgraph.append(None)
                    continue
                # Create initial graph as composition of subgraphs
                initial_graph = nx.Graph()
                for s in subgraph:
                    initial_graph = nx.compose(initial_graph,s)
                # Run simulation
                p, samples, idx, success = at.prob_dist(self.X,self.O,self.capacity,initial_graph=initial_graph,max_edges=True,max_iters=max_iters,rewire_est=True,multiedge=self.multiedge)
                p = p / np.sum(p)
                # Check whether sample is subgraph of G
                for j, s in enumerate(samples):
                    iso_object = nx.isomorphism.GraphMatcher(self.target,s)
                    if iso_object.subgraph_is_isomorphic() and p[j] > prob_tol:
                        node.data.p.append(p[j]*probs[i])
                        node.data.subgraph.append(s)
                        if success is False:
                            self.success = False
                        break
                    else:
                        node.data.p.append(0)
                        node.data.subgraph.append(s)
                        if success is False:
                            self.success = False

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
    def __init__(self, nodes, X, O, capacity, subgraph=[], p=[]):
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
        self.p = p
        self.count = len(nodes)
        self.success = np.nan

def convert_keys_to_int(d):
    """
    Recursively convert all keys in a nested dictionary to int, skipping keys that cannot be converted.
    
    Parameters
    ----------
    d : dict -- The nested dictionary.

    Returns
    -------
    dict -- The dictionary with keys converted to int where possible.
    """
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            try:
                new_key = int(k)
            except ValueError:
                new_key = k  # Keep the original key if it can't be converted
            new_dict[new_key] = convert_keys_to_int(v)
        return new_dict
    elif isinstance(d, list):
        return [convert_keys_to_int(item) for item in d]
    else:
        return d

def expand_tree(tree_dict):
    stack = [tree_dict]
    while stack:
        current = stack.pop()
        for node_id, node in current.items():
            if 'data' in node:
                node['data'] = np.sort(np.array(node['data'].nodes,dtype=int)).tolist()
            if 'children' in node:
                stack.extend(node['children'])
    return convert_keys_to_int(tree_dict)

class DesignMCMC:
    """
    DesignMCMC class for performing MCMC sampling on assembly trees.
    This class implements the Metropolis-Hastings algorithm to sample from the posterior distribution of assembly trees given a target graph and binding matrix.


    Attributes
    ----------
    T : numpy.ndarray -- The assembly tree.
    proposed_T : numpy.ndarray -- The proposed assembly tree for the next iteration.
    cur_prob : float -- The current log probability of the assembly tree.
    samples : list -- List of sampled assembly trees.
    log_p : list -- List of log probabilities of the sampled assembly trees.
    dist : numpy.ndarray -- Estimated probability distribution
    uniq_samples : list -- List of unique assembly trees sampled.
    """

    def __init__(self, T):
        """
        Initialize the DesignMCMC class with the assembly tree, target graph, node labels, and binding matrix.
        """
        self.cur_T = T
        self.proposed_T = copy.copy(T)
        self.cur_prob = copy.deepcopy(np.log(sum(self.cur_T.Tree.get_node(0).data.p)))
        self.samples = []
        self.log_p = []
        self.unique_samples = []
        self.dist = []
        pass
    
    def run_mcmc(self, num_samples, prior='uniform',verbose=False):
        """
        Run the MCMC sampling algorithm.
        
        Parameters
        ----------
        num_samples : int -- Number of samples to generate.

        Returns
        -------
        samples : list -- List of sampled assembly trees.
        """
        if not verbose:
            for i in range(num_samples):
                # Propose a new tree
                update_success = False
                while update_success == False:
                    update_success = self.proposed_T.update_tree()

                # Get prior of tree
                if prior == 'depth':
                    prior_val = self.tree_prior(self.proposed_T)
                # Uniform prior
                if prior == 'uniform':
                    prior_val = 0
                # Get likelihood of tree
                p = sum(self.proposed_T.Tree.get_node(0).data.p)
                likelihood_val = np.log(p) if p > 10e-300 else np.log(10e-300) # Avoid log(0) by using a very small value
                # Calculate new posterior log prob
                posterior = prior_val + likelihood_val
                # Calculate acceptance probability
                acceptance_prob = np.min([1, np.exp(posterior - self.cur_prob)])
                # Print acceptance probability, prior_val and likelihood_val
                # print(f"Iteration {i+1}/{num_samples}, Acceptance Probability: {acceptance_prob:.4f}, Prior: {prior_val:.4f}, Likelihood: {likelihood_val:.4f}, Previous Posterior: {self.cur_prob:.4f}, New Posterior: {posterior:.4f}")
                # Update current tree and probability if accepted
                if np.random.rand() < acceptance_prob:
                    self.cur_T = copy.deepcopy(self.proposed_T)
                    self.cur_prob = posterior
                    self.samples.append(copy.deepcopy(self.cur_T))
                    self.log_p.append(posterior)
                else:
                    # If not accepted, revert to the current tree
                    self.proposed_T = copy.deepcopy(self.cur_T)
                    self.samples.append(copy.deepcopy(self.cur_T))
                    self.log_p.append(copy.deepcopy(self.cur_prob))
        else:
            for i in tqdm(range(num_samples)):
                # Propose a new tree
                update_success = False
                while update_success == False:
                    update_success = self.proposed_T.update_tree()

                # Get prior of tree
                if prior == 'depth':
                    prior_val = self.tree_prior(self.proposed_T)
                # Uniform prior
                if prior == 'uniform':
                    prior_val = 0
                # Get likelihood of tree
                p = sum(self.proposed_T.Tree.get_node(0).data.p)
                likelihood_val = np.log(p) if p > 0 else np.log(10e-300) # Avoid log(0) by using a small value
                # Calculate new posterior log prob
                posterior = prior_val + likelihood_val
                # Calculate acceptance probability
                acceptance_prob = np.min([1, np.exp(posterior - self.cur_prob)])
                # Print acceptance probability, prior_val and likelihood_val
                # print(f"Iteration {i+1}/{num_samples}, Acceptance Probability: {acceptance_prob:.4f}, Prior: {prior_val:.4f}, Likelihood: {likelihood_val:.4f}, Previous Posterior: {self.cur_prob:.4f}, New Posterior: {posterior:.4f}")
                # Update current tree and probability if accepted
                if np.random.rand() < acceptance_prob:
                    self.cur_T = copy.deepcopy(self.proposed_T)
                    self.cur_prob = posterior
                    self.samples.append(copy.deepcopy(self.cur_T))
                    self.log_p.append(posterior)
                else:
                    # If not accepted, revert to the current tree
                    self.proposed_T = copy.deepcopy(self.cur_T)
                    self.samples.append(copy.deepcopy(self.cur_T))
                    self.log_p.append(copy.deepcopy(self.cur_prob))
        # Update the estimated probability distribution of the assembly tree
        self.update_dist()
    
    def update_dist(self):
        """
        Update the estimated probability distribution of the assembly tree.
        """
        # Get all unique assembly trees
        self.unique_samples = []
        self.dist = []
        count = []
        for i, s in enumerate(self.samples):
            tree = s.Tree.to_dict(with_data=True)
            expand_tree(tree)
            new = True
            for j, k in enumerate(self.unique_samples):
                if k == s:
                    count[k] += 1
                    self.dist[k] += self.log_p[i]
                    new = False
                    break
            if new:
                self.unique_samples.append(tree)
                count.append(1)
                self.dist.append(self.log_p[i])
        self.dist = np.array(self.dist)
        count = np.array(count)
        self.dist /= count
    
    def recursive_summation(self,depth, current_Q):
        if depth <= 2:
            return 1
        if depth == 2:
            return stirling2(current_Q, 2)
        summation = 0
        for an in range(depth, current_Q - 1):
            s = stirling2(current_Q, an)
            summation += s*self.recursive_summation(depth - 1, an)
        return summation

    def tree_prior(self,T):
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
        D = T.Tree.depth()
        # Get number of partitions
        Q = len(T.Tree.leaves())

        # Compute Bell Number on N nodes
        N = T.X.shape[0]
        bell_number = 0
        for k in range(N + 1):
            bell_number += stirling2(N, k)
        
        # Calculate internal count of number of possible splits with depth D
        num_trees = self.recursive_summation(D, Q)

        return - np.log(bell_number) - np.log(Q) - np.log(num_trees)

