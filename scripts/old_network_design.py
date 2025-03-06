import networkx as nx
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
from itertools import product
import copy
from scipy.spatial.distance import cdist
# try:
import graph_tool.all as gt

def compute_probability_matrix(N_vec, O, X, directed=False):
    """
    Compute the probability matrix for network connections.

    Parameters:
        N_vec (ndarray): Distribution of particle types.
        O (ndarray): Connection matrix.
        X (ndarray): Node assignment matrix (one-hot encoded).
        directed (bool): Whether the network is directed.

    Returns:
        P (ndarray): Probability matrix.
    """
    N = len(X)
    P = np.zeros((N, N))
    labels = np.argmax(X, axis=1)  # Node labels

    for i in range(N):
        for j in (range(N) if directed else range(i, N)):
            theta_i, theta_j = labels[i], labels[j]
            if directed:
                P[i, j] = O[theta_i, theta_j] / N_vec[theta_j]
            else:
                prob_ij = O[theta_i, theta_j] / N_vec[theta_j]
                prob_ji = O[theta_j, theta_i] / N_vec[theta_i]
                P[i, j] = min(1, min(prob_ij, prob_ji))
    return P


def probability(A, N_vec, O, X, directed=False):
    """
    Calculate the probability of a given adjacency matrix.

    Parameters:
        A (ndarray): Adjacency matrix.
        N_vec (ndarray): Distribution of particle types.
        O (ndarray): Connection matrix.
        X (ndarray): Node assignment matrix (one-hot encoded).
        directed (bool): Whether the network is directed.

    Returns:
        float: The probability of the adjacency matrix.
    """
    N = A.shape[0]
    P = compute_probability_matrix(N_vec, O, X, directed)
    prob = np.prod(P[A == 1]) * np.prod(1 - P[A == 0])
    return prob


def canonical_ensemble(N_vec: np.ndarray, O: np.ndarray, X: np.ndarray, directed=False, ret_P=False):
    """
    Generate a network from the canonical ensemble.

    Parameters:
        N_vec (ndarray): Distribution of particle types.
        O (ndarray): Connection matrix.
        X (ndarray): Node assignment matrix (one-hot encoded).
        directed (bool): Whether the network is directed.
        ret_P (bool): Whether to return the probability matrix.

    Returns:
        nx.Graph or nx.DiGraph: Generated network.
        (optional) ndarray: Probability matrix if ret_P is True.
    """
    N = len(X)
    P = compute_probability_matrix(N_vec, O, X, directed)
    r = np.random.random(size=(N, N))
    A = (r < P).astype(int)

    if not directed:
        A = np.triu(A) + np.triu(A, k=1).T  # Symmetrize adjacency matrix

    graph_type = nx.DiGraph if directed else nx.Graph
    G = nx.from_numpy_array(A, create_using=graph_type)

    return (G, P) if ret_P else G
    
def assign_particles(N, k_vals):
    """
    Assign N distinct particles, each with a specified degree.

    Parameters:
        N (int): Number of particles.
        k_vals (int or ndarray): Degrees of the particles. Can be a scalar or an array of size N.

    Returns:
        dict: A dictionary where keys are particle indices and values are dictionaries
              representing connections and their frequencies.
    """
    # Ensure k_vals is an array
    k_vals = np.full(N, k_vals, dtype=int) if isinstance(k_vals, int) else np.asarray(k_vals, dtype=int)
    
    # Generate all connections at once
    connections = [np.random.choice(np.arange(N), size=k, replace=True) for k in k_vals]
    
    # Build capacity dictionary
    capacity = {
        i: {j: np.sum(connections[i] == j) for j in np.unique(connections[i])}
        for i in range(N)
    }
    
    return capacity


def create_O(capacities):
    """
    Create the O matrix from capacities.

    Parameters:
        capacities (dict): A dictionary where keys are particle indices, and values are dictionaries 
                           of connections and their frequencies.

    Returns:
        np.ndarray: The O matrix.
    """
    N = len(capacities)
    O = np.zeros((N, N), dtype=float)
    
    for i, connections in capacities.items():
        for j, value in connections.items():
            O[i, j] = value
    
    return O

def generate_node_assignments(N, k):
    """
    Create a random X matrix for N particles, each belonging to one of k classes.

    Parameters:
        N (int): Number of particles (rows in the matrix).
        k (int): Number of classes (columns in the matrix).

    Returns:
        np.ndarray: A one-hot encoded matrix of shape (N, k).
    """
    X = np.zeros((N, k), dtype=int)
    X[np.arange(N), np.random.randint(0, k, size=N)] = 1
    return X


# def soup_of_nodes(X,capacities,T=10000,ret_H=False,time_of_entry=None,self_loops=True, ret_cap=False):
#     # initialize network
#     if self_loops:
#         g = nx.MultiGraph()
#     else:
#         g = nx.Graph()
#     if time_of_entry is None:
#         time_of_entry = np.zeros(len(X))
    
#     # Add node to network
#     for t in range(T):
#         node_idx = np.sort(np.where(time_of_entry == t)[0])
#         g.add_nodes_from(node_idx)
#         # Randomly select two nodes
#         node0, node1 = np.random.choice(list(g.nodes()),size=2,replace=False)
            
#         # Get current node label
#         label0 = int(np.where(X[node0] == 1)[0])
#         label1 = int(np.where(X[node1] ==1)[0])
        
#         edge0_capacity = capacities[node0][label1]
#         edge1_capacity = capacities[node1][label0]
#         if edge0_capacity <= 0 or edge1_capacity <= 0:
#             continue
        
#         else:
#             if self_loops is False:
#                 if g.has_edge(node0,node1) is False:
#                     g.add_edge(node0,node1)
                
#             # Update capacities
#             capacities[node0][label1] -= 1
#             capacities[node1][label0] -= 1
#     if ret_cap:
#         return g, capacities
#     return g

# def microcanonical_ensemble(X,capacities,ret_H=False,directed=False,time_of_entry=None,self_loops=True, ret_cap=False):
#     # initialize network
#     if self_loops:
#         g = nx.MultiGraph()
#     elif directed:
#         g = nx.DiGraph()
#     else:
#         g = nx.Graph()
#     if time_of_entry is None:
#         time_of_entry = np.zeros(len(X))
    
#     # Order node pairs randomly
#     g.add_nodes_from(np.arange(len(X)))
    
#     if directed:
#         node_pairs = np.array(list(permutations(g.nodes(),2)))
#     else:
#         node_pairs = np.array(list(combinations(g.nodes(),2)))
#     np.random.shuffle(node_pairs)
#     # Add node to network
#     for node0, node1 in node_pairs:
            
#         # Get current node label
#         label0 = int(np.where(X[node0] == 1)[0])
#         label1 = int(np.where(X[node1] ==1)[0])
        
#         edge0_capacity = capacities[node0][label1]
#         edge1_capacity = capacities[node1][label0]
        
#         if directed:
#             if edge0_capacity <= 0:
#                 continue
#             else:
#                 if self_loops is False:
#                     if g.has_edge(node0,node1) is False:
#                         g.add_edge(node0,node1)
                
#                 capacities[node0][label1] -= 1
#         elif edge0_capacity <= 0 or edge1_capacity <= 0:
#             continue
        
#         else:
#             if self_loops is False:
#                 if g.has_edge(node0,node1) is False:
#                     g.add_edge(node0,node1)
#             else:
#                 g.add_edge(node0,node1)
                
#             # Update capacities
#             capacities[node0][label1] -= 1
#             capacities[node1][label0] -= 1
#     if ret_cap:
#         return g, capacities
#     return g


def microcanonical_ensemble(
    X,
    O,
    ret_H = False,
    directed = False,
    time_of_entry = None,
    multiedge = False,
    ret_cap = False
):
    """
    Generate a microcanonical ensemble for network design.

    Parameters:
        X (ndarray): Node assignment matrix (one-hot encoded).
        capacities (dict): Connection capacities for each node and label.
        ret_H (bool): Reserved parameter (currently unused).
        directed (bool): Whether the network is directed.
        time_of_entry (ndarray, optional): Entry times of nodes. Defaults to all zeros.
        self_loops (bool): Whether self-loops are allowed.
        ret_cap (bool): Whether to return updated capacities.

    Returns:
        nx.Graph or tuple: Generated graph, optionally with updated capacities.
    """
    # Initialize graph
    graph_type = nx.MultiGraph if multiedge else (nx.DiGraph if directed else nx.Graph)
    g = graph_type()
    g.add_nodes_from(range(len(X)))
    
    if time_of_entry is None:
        time_of_entry = np.zeros(len(X), dtype=float)

    # Generate and shuffle node pairs
    if directed:
        node_pairs = np.array(list(permutations(g.nodes, 2)))
    else:
        node_pairs = np.array(list(combinations(g.nodes, 2)))
    np.random.shuffle(node_pairs)
    capacities = create_capacity(X,O)
    # Process node pairs
    labels = np.argmax(X, axis=1)  # Precompute node labels
    for node0, node1 in node_pairs:
        if not multiedge and node0 == node1:
            continue

        label0, label1 = labels[node0], labels[node1]
        edge0_capacity = capacities[node0].get(label1, 0)
        edge1_capacity = capacities[node1].get(label0, 0)

        if directed:
            if edge0_capacity > 0:
                g.add_edge(node0, node1)
                capacities[node0][label1] -= 1
        elif edge0_capacity > 0 and edge1_capacity > 0:
            g.add_edge(node0, node1)
            capacities[node0][label1] -= 1
            capacities[node1][label0] -= 1

    return (g, capacities) if ret_cap else g



# def link_soup(X, edge_capacities, ret_H=False, directed=False, time_of_entry=None, self_loops=False, ret_cap=False):
#     # Initialize network
#     g = nx.MultiGraph() if self_loops else (nx.DiGraph() if directed else nx.Graph())
#     g.add_nodes_from(np.arange(len(X)))
    
#     if time_of_entry is None:
#         time_of_entry = np.zeros(len(X))

#     # Precompute node labels to avoid repeated calculations
#     labels = np.argmax(X, axis=1)

#     # Get all node pairs
#     node_pairs = (np.array(list(permutations(g.nodes(), 2))) if directed else np.array(list(combinations(g.nodes(), 2))))
#     np.random.shuffle(node_pairs)

#     # Process each node pair
#     for node0, node1 in node_pairs:
#         if node0 == node1:
#             continue
#         if g.has_edge(node0,node1):
#             continue
            
#         label0, label1 = labels[node0], labels[node1]

#         # Get viable links
#         links0 = [link for link in edge_capacities[node0] if label1 in np.array([edge_capacities[node0][link]])]
#         links1 = [link for link in edge_capacities[node1] if label0 in np.array([edge_capacities[node1][link]])]
#         if not links0 or not links1:
#             continue

#         # Choose random links from viable options
#         link0, link1 = np.random.choice(links0), np.random.choice(links1)
#         # Update edge capacities
#         edge_capacities[node0][link0] = ()
#         edge_capacities[node1][link1] = ()

#         # # Add edge to the graph
#         # if not self_loops and node0 == node1:
#         #     continue
        
#         g.add_edge(node0, node1)

#     return (g, edge_capacities) if ret_cap else g

def extract_capacities(g: nx.Graph, X: np.ndarray) -> dict:
    """
    Extract capacities from a network based on node labels.

    Parameters:
        g (nx.Graph): The network.
        X (np.ndarray): One-hot encoded label matrix for nodes.

    Returns:
        dict: A dictionary where keys are labels and values are capacities for each label.
    """
    num_labels = X.shape[1]
    
    # Initialize capacity dictionary
    capacity = {i: {j: 0 for j in range(num_labels)} for i in range(num_labels)}
    
    # Iterate through nodes to calculate capacities
    for node in g.nodes:
        # Get node's label
        node_label = np.argmax(X[node])
        
        # Get neighbors' labels
        neighbors = list(g.neighbors(node))
        if neighbors:  # Skip nodes with no neighbors
            neighbor_labels = X[neighbors].sum(axis=0)
            
            # Update capacity dictionary
            for j, count in enumerate(neighbor_labels):
                capacity[node_label][j] = max(capacity[node_label][j], count)
    
    return capacity


# def rgg_network_design(X,pos,capacities):
#     # initialize network
#     g = nx.MultiGraph()
#     g.add_nodes_from(np.arange(len(X)))
    
#     # if time_of_entry is None:
#     #     time_of_entry = np.zeros(len(X))
#     dist = np.triu(cdist(pos,pos))
#     idx0, idx1 = np.triu_indices(dist.shape[0],k=1)
#     sorted_idx = np.argsort(dist[np.triu_indices(dist.shape[0],k=1)])
    
#     # Add node to network
#     for t in range(len(sorted_idx)):
#         idx = np.where(sorted_idx==t)[0][0]
#         node0, node1 = idx0[idx], idx1[idx]
#         # Randomly select two nodes
            
            
#         # Get current node label
#         label0 = int(np.where(X[node0] == 1)[0])
#         label1 = int(np.where(X[node1] ==1)[0])
        
#         edge0_capacity = capacities[node0][label1]
#         edge1_capacity = capacities[node1][label0]
        
#         if edge0_capacity <= 0 or edge1_capacity <= 0:
#             continue
#         else:
#             g.add_edge(node0,node1)
                
#             # Update capacities
#             capacities[node0][label1] -= 1
#             capacities[node1][label0] -= 1

#     return g



def create_capacity(X: np.ndarray, O: np.ndarray) -> dict:
    """
    Create a capacity dictionary based on node labels.

    Parameters:
        X (np.ndarray): One-hot encoded label matrix for nodes.
        O (np.ndarray): Adjacency-like matrix where O[i, j] indicates if a node of type i can connect to type j.

    Returns:
        dict: Capacities for each node.
    """
    capacities = {}
    node_labels = np.argmax(X, axis=1)  # Precompute labels for all nodes
    
    for i, label in enumerate(node_labels):
        # Use a dictionary comprehension to set capacities based on `O`
        capacities[i] = {k: int(O[label, k]) for k in range(O.shape[1])}
    
    return capacities



# class NetDesign:
#     """
#     Creates a network using our network design framework.
    
#     Attributes:
#         g (nx.Graph) - network
#         X (ndarray) - matrix of node labels
#         capacity (dict) - capacities of each particle type
#         node_capacity (dict) - capacitys of each node
#         time_of_entry (ndarray) - array of time of entry for each node
#     """
#     def __init__(self,X,capacity,time_of_entry=None):
#         self.X = X
#         self.capacity = capacity
        
#         # Create node capacity dictionary
#         self.node_capacity = create_capacity(self.X,self.capacity)

#         # Initialize network
#         self.g = nx.MultiGraph()
        
#         # # Initialize H
#         # self.create_H()
        
#         # Initialize time_of_entry
#         if time_of_entry is None:
#             self.time_of_entry = np.zeros(len(self.X))
#         else:
#             self.time_of_entry = time_of_entry
            
#     # def create_H(self):
#     #     """
#     #     Create H matrix based on node capacities.
#     #     """
#     #     # Initialize H
#     #     self.H = np.zeros((len(self.X),len(self.X)))
#     #     for i in range(len(self.X)):
#     #         for j in range(i+1, len(self.X)):
#     #             # Get possible connections
#     #             label_i = int(np.where(self.X[i] == 1)[0])
#     #             label_j = int(np.where(self.X[j] == 1)[0])
#     #             self.H[i,j] = min(self.node_capacity[i][label_j],self.node_capacity[j][label_i])
#     #             self.H[j,i] = self.H[i,j]
#     #     pass
    
# #     def update_H(self,i,j):
# #         """
# #         Update H matrix based on node capacities
        
# #         Parameters:
# #             i (int) - node
# #             j (int) - node
# #         """
# #         # Get node labels
# #         label_i = int(np.where(self.X[i] == 1)[0])
# #         label_j = int(np.where(self.X[j] == 1)[0])
        
# #         # Get nodes with label i0
# #         nodes_with_label_i = np.where(self.X[:,label_i] == 1)[0]
# #         nodes_with_label_j = np.where(self.X[:,label_j] == 1)[0]
        
# #         for k in nodes_with_label_i:
# #             self.H[k,j] = min(self.node_capacity[j][label_i],self.node_capacity[k][label_j])
# #             self.H[j,k] = self.H[k,j]
            
# #         for k in nodes_with_label_j:
# #             self.H[k,i] = min(self.node_capacity[i][label_j],self.node_capacity[k][label_i])
# #             self.H[i,k] = self.H[k,i]
        
# #         pass
        
#     def reset(self):
#         """
#         Reset simulation.
#         """
#         # Create node capacity dictionary
#         self.node_capacity = create_capacity(self.X,self.capacity)
        
#         # Initialize network
#         self.g = nx.MultiGraph()
        
# #         # Initialize H
# #         self.H = np.zeros((len(self.X),len(self.X)))
# #         for i in range(len(self.X)):
# #             for j in range(i+1, len(self.X)):
# #                 # Get possible connections
# #                 label_i = int(np.where(self.X[i] == 1)[0])
# #                 label_j = int(np.where(self.X[j] == 1)[0])
                
# #                 self.H[i,j] = min(self.node_capacity[i][label_j],self.node_capacity[j][label_i])
# #                 self.H[j,i] = self.H[i,j]
                
#     def simulate(self,T,r=None,kmax=None,ret_cap=False,multilinks=False):
#         """
#         Run simulation for T timesteps where each pair of nodes interact with probability r at each time step.
        
#         Parameters:
#             T (int) - number of time steps
#             r (float or ndarray) - rate at which interactions between nodes occur. Assumed to be the probability that the nodes interact at the time step. Random interactions if None.
#             ret_H (bool) - return array of H over time.
#         """
#         # Save H
#         if ret_cap:
#             all_cap = {}
#             all_cap[0] = copy.deepcopy(self.node_capacity)
            
#         # Check whether r is float
#         if type(r) == float:
#             r *= np.ones((len(self.X),len(self.X)))
        
            
#         # Initialize simulation
#         for t in range(T):
#             # Add new nodes based on time of entry
#             node_idx = np.sort(np.where(self.time_of_entry == t)[0])
#             self.g.add_nodes_from(node_idx)
            
#             if r is None:
#                 N = self.g.number_of_nodes()
#                 r = (2 / (N*(N-1))) * np.ones((len(self.X),len(self.X)))
                
#             # Draw simulation from r
#             random_draw = np.random.random(size=r.shape)
            
#             # Create interaction matrix
#             I = random_draw < r

#             # Get interactions
#             interactions0, interactions1 = np.where(I == True)
            
#             # Randomize order of interactions
#             idx = np.arange(len(interactions0))
#             np.random.shuffle(idx)
            
#             # Connect particles
#             for i, j in zip(interactions0[idx],interactions1[idx]):
#                 # Skip if same node
#                 if i == j:
#                     continue
#                 if multilinks:
#                     if self.g.has_edge(i,j):
#                         continue
                    
#                 # Check that both nodes are in network
#                 if self.g.has_node(i) and self.g.has_node(j):
#                     # Get node labels
#                     label_i = int(np.where(self.X[i] == 1)[0])
#                     label_j = int(np.where(self.X[j] == 1)[0])
                    
#                     # Check that nodes have capacity
#                     if self.node_capacity[i][label_j] > 0 and self.node_capacity[j][label_i] > 0:
#                         if kmax is not None:
#                             if self.g.degree(i) == kmax[i] or self.g.degree(j) == kmax[j]:
#                                 continue
                                
#                         # Connect nodes
#                         self.g.add_edge(i,j)
                        
#                         # Update node_capacities
#                         self.node_capacity[i][label_j] -= 1
#                         self.node_capacity[j][label_i] -= 1

                        
                        
#                         # # Update H
#                         # self.update_H(i,j)
                        
#             # Save node_capacity
#             if ret_cap:
#                 all_cap[t+1] = copy.deepcopy(self.node_capacity)
#         if ret_cap:
#             return all_cap
#         else:
#             pass
                        
class NetDesign:
    """
    Creates a network using a network design framework.
    
    Attributes:
        g (nx.Graph): The network.
        X (ndarray): Matrix of node labels (one-hot encoded).
        capacity (dict): Capacities for each particle type.
        node_capacity (dict): Capacities for each node.
        time_of_entry (ndarray): Array representing time of entry for each node.
    """
    def __init__(self, X, capacity, time_of_entry=None):
        self.X = X
        self.capacity = capacity
        
        # Create node capacity dictionary
        self.node_capacity = create_capacity(self.X, self.capacity)

        # Initialize network
        self.g = nx.MultiGraph()
        
        # Initialize time_of_entry
        self.time_of_entry = (
            np.zeros(len(self.X)) if time_of_entry is None else time_of_entry
        )
            
    def reset(self):
        """
        Reset the simulation by reinitializing node capacities and the network.
        """
        self.node_capacity = create_capacity(self.X, self.capacity)
        self.g = nx.MultiGraph()
        
    def simulate(self, T, r=None, kmax=None, ret_cap=False, multilinks=False):
        """
        Run the simulation for T timesteps, where node pairs interact probabilistically.
        
        Parameters:
            T (int): Number of timesteps.
            r (float or ndarray): Interaction probability matrix. Random interactions if None.
            kmax (ndarray): Maximum degree for each node.
            ret_cap (bool): Whether to return node capacities over time.
            multilinks (bool): If True, disallow multiple edges between node pairs.
        
        Returns:
            dict (optional): Node capacities over time if ret_cap is True.
        """
        # Prepare to store capacities over time
        if ret_cap:
            all_cap = {0: self.node_capacity.copy()}

        # Create interaction probability matrix if not provided
        if r is None:
            N = len(self.X)
            r = (2 / (N * (N - 1))) * np.ones((N, N))
        elif isinstance(r, float):
            r = r * np.ones((len(self.X), len(self.X)))

        # Simulation loop
        for t in range(T):
            # Add new nodes entering at this timestep
            node_idx = np.flatnonzero(self.time_of_entry == t)
            self.g.add_nodes_from(node_idx)
            
            # Draw random interactions
            random_draw = np.random.random(r.shape)
            interaction_matrix = random_draw < r
            
            # Get interaction pairs
            interactions = np.column_stack(np.where(interaction_matrix))
            np.random.shuffle(interactions)  # Randomize interaction order
            
            # Process interactions
            for i, j in interactions:
                # Skip invalid interactions
                if i == j or (multilinks and self.g.has_edge(i, j)):
                    continue
                if not (self.g.has_node(i) and self.g.has_node(j)):
                    continue
                
                # Retrieve node labels
                label_i = np.argmax(self.X[i])
                label_j = np.argmax(self.X[j])
                
                # Check capacities and degree constraints
                if (
                    self.node_capacity[i][label_j] > 0 
                    and self.node_capacity[j][label_i] > 0
                    and (kmax is None or self.g.degree[i] < kmax[i] and self.g.degree[j] < kmax[j])
                ):
                    # Add edge and update capacities
                    self.g.add_edge(i, j)
                    self.node_capacity[i][label_j] -= 1
                    self.node_capacity[j][label_i] -= 1
            
            # Save node capacities if required
            if ret_cap:
                all_cap[t + 1] = self.node_capacity.copy()

        return all_cap if ret_cap else None
