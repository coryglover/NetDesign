import networkx as nx
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
from itertools import product
import copy
from scipy.spatial.distance import cdist

def probability(A,N_vec,O,X,directed=False):
    """
    Generate the canonical ensemble for network design.
    
    Parameters:
        N_vec (ndarray) - distribution of particle types
        O (ndarray) - connection matrix
        X (ndarray) - node assignment matrix (one hot encoded)
        directed (bool) - return a directed network
    
    Returns:
        g (nx.Graph or nx.DiGraph)
    """
    # Get number of nodes
    N = np.sum(N_vec)
    # Generate probability matrix
    P = np.zeros((N,N))
    if directed:
        for i in range(N):
            for j in range(N):
                # Get labels of nodes
                theta_i = np.where(X[i]==1)[0][0]
                theta_j = np.where(X[j]==1)[0][0]
                P[i,j] = O[theta_i,theta_j] / N_vec[theta_j]
    else:
        for i in range(N):
            for j in range(i,N):
                # Get labels of nodes
                theta_i = np.where(X[i]==1)[0][0]
                theta_j = np.where(X[j]==1)[0][0]
                P[i,j] = np.min([1,np.min([O[theta_i,theta_j]/N_vec[theta_j],O[theta_j,theta_i]/N_vec[theta_i]])])
                
    prob = np.prod([P[i,j]**(A[i,j])*(1-P[i,j])**(1-A[i,j]) for j in range(i,A.shape[0]) for i in range(A.shape[0])])
    
    return prob

def canonical_ensemble(N_vec,O,X,directed=False):
    """
    Generate the canonical ensemble for network design.
    
    Parameters:
        N_vec (ndarray) - distribution of particle types
        O (ndarray) - connection matrix
        X (ndarray) - node assignment matrix (one hot encoded)
        directed (bool) - return a directed network
    
    Returns:
        g (nx.Graph or nx.DiGraph)
    """
    # Get number of nodes
    N = np.sum(N_vec)
    # Generate probability matrix
    P = np.zeros((N,N))
    if directed:
        for i in range(N):
            for j in range(N):
                # Get labels of nodes
                theta_i = np.where(X[i]==1)[0][0]
                theta_j = np.where(X[j]==1)[0][0]
                P[i,j] = O[theta_i,theta_j] / N_vec[theta_j]
    else:
        for i in range(N):
            for j in range(i,N):
                # Get labels of nodes
                theta_i = np.where(X[i]==1)[0][0]
                theta_j = np.where(X[j]==1)[0][0]
                P[i,j] = np.min([1,np.min([O[theta_i,theta_j]/N_vec[theta_j],O[theta_j,theta_i]/N_vec[theta_i]])])
    
    # Generate random samples
    r = np.random.random(size=(N,N))
    # Compare
    idx = np.where(r < P)
    
    A = np.zeros((N,N))
    if directed:
        A[idx] = 1
    else:
        A[idx] = 1
        A = np.triu(A) + np.triu(A).T
        A[A>1] = 1
    
    if directed:
        return nx.from_numpy_array(A,create_using=nx.DiGraph)
    else:
        return nx.from_numpy_array(A)
    
def assign_particles(N,k_vals):
    """
    Create N distinct particles with each with degree k
    """
    capacity = {i:{} for i in range(N)}
    particle_list = np.arange(N)
    if type(k_vals) is int:
        k_vals = np.ones(N)*k_vals
        
    for i, k in zip(particle_list,k_vals):
        # Get connections
        connections = np.random.choice(particle_list,k,replace=True)
        capacity[i] = {j:len(np.where(connections==j)[0]) for j in range(N)}
    return capacity

def create_O(capacities):
    """
    Create O matrix from capacities
    """
    O = np.zeros((len(capacities.keys()),len(capacities.keys())))
    for i in capacities.keys():
        for j in capacities.keys():
            O[i,j] = capacities[i][j]
    return O

def generate_node_assignments(N,k):
    """
    Create random X matrix on N particles which each can be one of k different classes
    """
    X = np.zeros((N,k))
    for i in range(X.shape[0]):
        X[i,np.random.choice(np.arange(k))] = 1
    return X

def soup_of_nodes(X,capacities,T=10000,ret_H=False,time_of_entry=None,self_loops=True, ret_cap=False):
    # initialize network
    if self_loops:
        g = nx.MultiGraph()
    else:
        g = nx.Graph()
    if time_of_entry is None:
        time_of_entry = np.zeros(len(X))
    
    # Add node to network
    for t in range(T):
        node_idx = np.sort(np.where(time_of_entry == t)[0])
        g.add_nodes_from(node_idx)
        # Randomly select two nodes
        node0, node1 = np.random.choice(list(g.nodes()),size=2,replace=False)
            
        # Get current node label
        label0 = int(np.where(X[node0] == 1)[0])
        label1 = int(np.where(X[node1] ==1)[0])
        
        edge0_capacity = capacities[node0][label1]
        edge1_capacity = capacities[node1][label0]
        if edge0_capacity <= 0 or edge1_capacity <= 0:
            continue
        
        else:
            if self_loops is False:
                if g.has_edge(node0,node1) is False:
                    g.add_edge(node0,node1)
                
            # Update capacities
            capacities[node0][label1] -= 1
            capacities[node1][label0] -= 1
    if ret_cap:
        return g, capacities
    return g

def new_soup_of_nodes(X,capacities,ret_H=False,directed=False,time_of_entry=None,self_loops=True, ret_cap=False):
    # initialize network
    if self_loops:
        g = nx.MultiGraph()
    elif directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    if time_of_entry is None:
        time_of_entry = np.zeros(len(X))
    
    # Order node pairs randomly
    g.add_nodes_from(np.arange(len(X)))
    
    if directed:
        node_pairs = np.array(list(permutations(g.nodes(),2)))
    else:
        node_pairs = np.array(list(combinations(g.nodes(),2)))
    np.random.shuffle(node_pairs)
    # Add node to network
    for node0, node1 in node_pairs:
            
        # Get current node label
        label0 = int(np.where(X[node0] == 1)[0])
        label1 = int(np.where(X[node1] ==1)[0])
        
        edge0_capacity = capacities[node0][label1]
        edge1_capacity = capacities[node1][label0]
        
        if directed:
            if edge0_capacity <= 0:
                continue
            else:
                if self_loops is False:
                    if g.has_edge(node0,node1) is False:
                        g.add_edge(node0,node1)
                
                capacities[node0][label1] -= 1
        elif edge0_capacity <= 0 or edge1_capacity <= 0:
            continue
        
        else:
            if self_loops is False:
                if g.has_edge(node0,node1) is False:
                    g.add_edge(node0,node1)
            else:
                g.add_edge(node0,node1)
                
            # Update capacities
            capacities[node0][label1] -= 1
            capacities[node1][label0] -= 1
    if ret_cap:
        return g, capacities
    return g

def extract_capacities(g,X):
    """
    Extract rules from network with given labels
    
    Parameters:
        g (nx.Graph) - network
        X (nd.array) - label matrix
    
    Returns:
        capacity (dict)
    """
    # Initialize dictionary
    capacity = {i:{j:0 for j in range(X.shape[1])} for i in range(X.shape[1])}
    for node in g.nodes():
        # Get neighbors
        neighbors = np.array(list(g.neighbors(node)))
        # Get label matrix of neighbors
        neighbor_labels = X[neighbors]
        # Get count of connections
        label_counts = neighbor_labels.sum(axis=0)
        
        # Update dictionary
        node_label = np.where(X[node] == 1)[0][0]
        for j in range(X.shape[1]):
            capacity[node_label][j] = int(np.max([capacity[node_label][j],label_counts[j]]))
            # capacity[j][node_label] = capacity[node_label][j]
    
    return capacity

def rgg_network_design(X,pos,capacities):
    # initialize network
    g = nx.MultiGraph()
    g.add_nodes_from(np.arange(len(X)))
    
    # if time_of_entry is None:
    #     time_of_entry = np.zeros(len(X))
    dist = np.triu(cdist(pos,pos))
    idx0, idx1 = np.triu_indices(dist.shape[0],k=1)
    sorted_idx = np.argsort(dist[np.triu_indices(dist.shape[0],k=1)])
    
    # Add node to network
    for t in range(len(sorted_idx)):
        idx = np.where(sorted_idx==t)[0][0]
        node0, node1 = idx0[idx], idx1[idx]
        # Randomly select two nodes
            
            
        # Get current node label
        label0 = int(np.where(X[node0] == 1)[0])
        label1 = int(np.where(X[node1] ==1)[0])
        
        edge0_capacity = capacities[node0][label1]
        edge1_capacity = capacities[node1][label0]
        
        if edge0_capacity <= 0 or edge1_capacity <= 0:
            continue
        else:
            g.add_edge(node0,node1)
                
            # Update capacities
            capacities[node0][label1] -= 1
            capacities[node1][label0] -= 1

    return g



def create_capacity(X,capacity_vals):
    capacities = {}
    for i in range(len(X)):
        # Get node label
        node_label = int(np.where(X[i]==1)[0])

        capacities[i] = copy.deepcopy(capacity_vals[node_label])
    return capacities

class NetDesign:
    """
    Creates a network using our network design framework.
    
    Attributes:
        g (nx.Graph) - network
        X (ndarray) - matrix of node labels
        capacity (dict) - capacities of each particle type
        node_capacity (dict) - capacitys of each node
        time_of_entry (ndarray) - array of time of entry for each node
    """
    def __init__(self,X,capacity,time_of_entry=None):
        self.X = X
        self.capacity = capacity
        
        # Create node capacity dictionary
        self.node_capacity = create_capacity(self.X,self.capacity)

        # Initialize network
        self.g = nx.MultiGraph()
        
        # # Initialize H
        # self.create_H()
        
        # Initialize time_of_entry
        if time_of_entry is None:
            self.time_of_entry = np.zeros(len(self.X))
        else:
            self.time_of_entry = time_of_entry
            
    # def create_H(self):
    #     """
    #     Create H matrix based on node capacities.
    #     """
    #     # Initialize H
    #     self.H = np.zeros((len(self.X),len(self.X)))
    #     for i in range(len(self.X)):
    #         for j in range(i+1, len(self.X)):
    #             # Get possible connections
    #             label_i = int(np.where(self.X[i] == 1)[0])
    #             label_j = int(np.where(self.X[j] == 1)[0])
    #             self.H[i,j] = min(self.node_capacity[i][label_j],self.node_capacity[j][label_i])
    #             self.H[j,i] = self.H[i,j]
    #     pass
    
#     def update_H(self,i,j):
#         """
#         Update H matrix based on node capacities
        
#         Parameters:
#             i (int) - node
#             j (int) - node
#         """
#         # Get node labels
#         label_i = int(np.where(self.X[i] == 1)[0])
#         label_j = int(np.where(self.X[j] == 1)[0])
        
#         # Get nodes with label i0
#         nodes_with_label_i = np.where(self.X[:,label_i] == 1)[0]
#         nodes_with_label_j = np.where(self.X[:,label_j] == 1)[0]
        
#         for k in nodes_with_label_i:
#             self.H[k,j] = min(self.node_capacity[j][label_i],self.node_capacity[k][label_j])
#             self.H[j,k] = self.H[k,j]
            
#         for k in nodes_with_label_j:
#             self.H[k,i] = min(self.node_capacity[i][label_j],self.node_capacity[k][label_i])
#             self.H[i,k] = self.H[k,i]
        
#         pass
        
    def reset(self):
        """
        Reset simulation.
        """
        # Create node capacity dictionary
        self.node_capacity = create_capacity(self.X,self.capacity)
        
        # Initialize network
        self.g = nx.MultiGraph()
        
#         # Initialize H
#         self.H = np.zeros((len(self.X),len(self.X)))
#         for i in range(len(self.X)):
#             for j in range(i+1, len(self.X)):
#                 # Get possible connections
#                 label_i = int(np.where(self.X[i] == 1)[0])
#                 label_j = int(np.where(self.X[j] == 1)[0])
                
#                 self.H[i,j] = min(self.node_capacity[i][label_j],self.node_capacity[j][label_i])
#                 self.H[j,i] = self.H[i,j]
                
    def simulate(self,T,r=None,kmax=None,ret_cap=False,multilinks=False):
        """
        Run simulation for T timesteps where each pair of nodes interact with probability r at each time step.
        
        Parameters:
            T (int) - number of time steps
            r (float or ndarray) - rate at which interactions between nodes occur. Assumed to be the probability that the nodes interact at the time step. Random interactions if None.
            ret_H (bool) - return array of H over time.
        """
        # Save H
        if ret_cap:
            all_cap = {}
            all_cap[0] = copy.deepcopy(self.node_capacity)
            
        # Check whether r is float
        if type(r) == float:
            r *= np.ones((len(self.X),len(self.X)))
        
            
        # Initialize simulation
        for t in range(T):
            # Add new nodes based on time of entry
            node_idx = np.sort(np.where(self.time_of_entry == t)[0])
            self.g.add_nodes_from(node_idx)
            
            if r is None:
                N = self.g.number_of_nodes()
                r = (2 / (N*(N-1))) * np.ones((len(self.X),len(self.X)))
                
            # Draw simulation from r
            random_draw = np.random.random(size=r.shape)
            
            # Create interaction matrix
            I = random_draw < r

            # Get interactions
            interactions0, interactions1 = np.where(I == True)
            
            # Randomize order of interactions
            idx = np.arange(len(interactions0))
            np.random.shuffle(idx)
            
            # Connect particles
            for i, j in zip(interactions0[idx],interactions1[idx]):
                # Skip if same node
                if i == j:
                    continue
                if multilinks:
                    if self.g.has_edge(i,j):
                        continue
                    
                # Check that both nodes are in network
                if self.g.has_node(i) and self.g.has_node(j):
                    # Get node labels
                    label_i = int(np.where(self.X[i] == 1)[0])
                    label_j = int(np.where(self.X[j] == 1)[0])
                    
                    # Check that nodes have capacity
                    if self.node_capacity[i][label_j] > 0 and self.node_capacity[j][label_i] > 0:
                        if kmax is not None:
                            if self.g.degree(i) == kmax[i] or self.g.degree(j) == kmax[j]:
                                continue
                                
                        # Connect nodes
                        self.g.add_edge(i,j)
                        
                        # Update node_capacities
                        self.node_capacity[i][label_j] -= 1
                        self.node_capacity[j][label_i] -= 1

                        
                        
                        # # Update H
                        # self.update_H(i,j)
                        
            # Save node_capacity
            if ret_cap:
                all_cap[t+1] = copy.deepcopy(self.node_capacity)
        if ret_cap:
            return all_cap
        else:
            pass
                        
            