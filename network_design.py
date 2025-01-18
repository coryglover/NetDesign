import networkx as nx
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
from itertools import product
import copy
from scipy.spatial.distance import cdist
import graph_tool.all as gt
from numba import jit
import numba as nb

def get_automorphic_groups_nx(graph):
    """
    Retrieve groups of automorphic vertices in the graph using NetworkX.
    """
    # Initialize a graph matcher
    GM = nx.isomorphism.GraphMatcher(graph, graph)
    
    # Collect all automorphisms
    automorphisms = []
    for automorphism in GM.isomorphisms_iter():
        automorphisms.append(automorphism)
    
    # Initialize equivalence classes
    n = graph.number_of_nodes()
    groups = {i: {i} for i in range(n)}
    
    # Merge equivalent vertices based on automorphisms
    for auto in automorphisms:
        for v1, v2 in auto.items():
            groups[v1].add(v2)
            groups[v2].add(v1)
    
    # Extract unique groups
    unique_groups = {frozenset(group) for group in groups.values()}
    return [list(group) for group in unique_groups]

def create_labels(N_dist):
    """
    Create label matrix from particle distribution.

    Parameters:
        N_dist (ndarray) - number of nodes of each particle type

    Returns:
        X (ndarray) - label matrix
    """
    X = np.zeros((N_dist.sum(),len(N_dist)))
    
    start_row = 0
    for col_idx, rows_in_group in enumerate(N_dist):
        X[start_row:start_row + rows_in_group,col_idx] = 1
        start_row += rows_in_group
        
    return X

def label_network(G, label_type='unique'):
    """
    Create labels for nodes of a network based on a predefined label type.

    Parameters:
        G (networkx.Graph): Input graph.
        label_type (str): 
            - 'unique': Each node receives a unique label.
            - 'automorphic': Nodes in the same automorphic group share the same label.

    Returns:
        np.ndarray: Label matrix (one-hot encoding).
    """
    num_nodes = G.number_of_nodes()
    
    if label_type == 'unique':
        # Each node gets a unique label (identity matrix)
        return np.eye(num_nodes, dtype=int)
    
    elif label_type == 'automorphic':
        # Automorphic grouping
        automorphic_groups = get_automorphic_groups_nx(G)
        
        # Map each node to its group index
        node_to_group = {node: group_idx 
                         for group_idx, group in enumerate(automorphic_groups)
                         for node in group}
        
        # Create a label matrix
        X = np.zeros((num_nodes, len(automorphic_groups)), dtype=int)
        
        for node, group_idx in node_to_group.items():
            X[node, group_idx] = 1
        
        return X
    
    else:
        raise ValueError("Invalid label_type. Choose 'unique' or 'automorphic'.")

def extract_O(g,X):
    """
    Extract binding matrix from network G and label matrix X.

    Parameters:
        g (networkx) - Graph
        X (ndarray) - label matrix

    Returns:
        O (ndarray) - binding matrix
    """
    # Get number of particle types
    particle_num = X.shape[1]
    labels = X.argmax(axis=1)

    # Initialize binding matrix
    O = np.zeros((particle_num,particle_num))

    # Iterate through each node and its neighbors
    for node in g.nodes:
        # Get node label
        node_label = labels[node]
        # Get neighbor labels
        neighbor_labels = labels[list(g.neighbors(node))]
        # Count neighbor labels
        counts = np.bincount(neighbor_labels, minlength=particle_num)
        # Update O matrix
        O[node_label] = np.maximum(O[node_label], counts)

    return O

def simulate(X,O,T=100,detachment_rate=.2,directed=False):
    """
    Simulate nodes interacting to form a network.
    Each node has a labeled defined by the matrix X, where X_{ij}=1 if node i has particle type j.
    Each particle type has binding rules encapsulated in O, where O_{ij} is the number of possible 
    connections a particle i can have with a particle j.
    Disconnected components in the network interact at a rate dependent on the number of nodes in the component.
    When they interact, the two components try to connect as much as possible.
    At each step, unstable nodes (those not full bound according to O) detach from their connected component at a rate 'detachment_rate'.
    We assume an undirected network.

    Parameters:
        X (ndarary) - label matrix
        O (ndarray) - binding matrix
        t (int) - number of time steps
        detachment_rate (float) - rate at which nodes disconnect

    Returns:
        g (networkx) - final graph
    """
    # Get labels of nodes
    labels = X.argmax(axis=1)

    # Get number of nodes
    N = len(labels)

    # Initialize network
    g = nx.Graph()
    g.add_nodes_from(range(N), dtype=int)

    # Simulate for T timesteps
    for t in range(T):
        # Select two random nodes
        n1, n2 = np.random.choice(N, size=2, replace=False)

        # Get components of the nodes
        c1 = nx.node_connected_component(g, n1)
        c2 = nx.node_connected_component(g, n2)

        if c1 == c2:
            continue  # Skip if nodes are already in the same component

        # Prepare connection attempts
        nodes_in_c1 = np.array(list(c1))
        nodes_in_c2 = np.array(list(c2))
        np.random.shuffle(nodes_in_c1)
        np.random.shuffle(nodes_in_c2)
        
        # Node pairs and viability check
        viable_pairs = []
        possible_links = []
        for e1 in nodes_in_c1:
            for e2 in nodes_in_c2:
                label1, label2 = labels[e1], labels[e2]
    
                # Check connection viability
                if (sum(labels[neighbor] == label2 for neighbor in g.neighbors(e1)) < O[label1, label2] and
                    sum(labels[neighbor] == label1 for neighbor in g.neighbors(e2)) < O[label2, label1]):
                    viable_pairs.append((e1, e2))
                if O[label1,label2] > 0 and O[label2,label1] > 0:
                    possible_links.append((e1,e2))
                    
        # Add viable edges
        if set(viable_pairs) == set(possible_links):
            g.add_edges_from(viable_pairs)

        # Batch detachment check
        detach_prob = np.random.random(size=N)
        nodes_to_check = np.where(detach_prob < detachment_rate)[0]

        for node in nodes_to_check:
            neighbors = list(g.neighbors(node))
            nlabels = labels[neighbors]

            # Check for under-connected nodes
            if any(nlabels.tolist().count(j) < O[labels[node], j] for j in range(O.shape[1])):
                g.remove_edges_from((node, neigh) for neigh in neighbors)

    return g

@jit(nopython=True, parallel=True)
def compute_viable_pairs_and_possible_links(comp1, comp2, X, O, neighbors, offsets, viable_pairs, possible_pairs):
    """
    Compute viable pairs and possible links between two components.

    Parameters:
        comp1 (array): Indices of nodes in component 1.
        comp2 (array): Indices of nodes in component 2.
        X (array): Label matrix for nodes.
        O (array): Binding matrix.
        offsets (array): Offset indices for neighbors in adjacency list.
    Returns:
        viable_pairs (list): List of viable pairs of nodes to connect.
        possible_links (list): List of all possible links between the two components.
    """
    counter = 0
    possible_counter = 0

    for u in nb.prange(len(comp1)):
        e1 = comp1[u]
        for v in nb.prange(len(comp2)):
            e2 = comp2[v]
            
            # Determine the labels of e1 and e2
            label1 = X[e1].argmax()
            label2 = X[e2].argmax()
            
            # Count how many neighbors of e1 have label2
            count1 = 0
            for i in nb.prange(offsets[e1], offsets[e1 + 1]):
                neighbor = neighbors[i]
                if X[neighbor, label2] == 1:
                    count1 += 1
            
            # Count how many neighbors of e2 have label1
            count2 = 0
            for i in nb.prange(offsets[e2], offsets[e2 + 1]):
                neighbor = neighbors[i]
                if X[neighbor, label1] == 1:
                    count2 += 1
            
            # Check connection viability
            if count1 < O[label1, label2] and count2 < O[label2, label1]:
                viable_pairs[counter] = (e1, e2)
                counter += 1
            
            # Add to possible links if O allows connections
            # if O[label1, label2] > 0 and O[label2, label1] > 0:
            #     possible_pairs[possible_counter] = (e1,e2)
            #     possible_counter += 1

    return viable_pairs,counter

# @jit(nopython=True)
# def compute_viable_pairs_and_possible_links(comp1, comp2, X, O, A, viable_pairs):
#     """
#     Compute viable pairs and possible links between two components.

#     Parameters:
#         comp1 (array): Indices of nodes in component 1.
#         comp2 (array): Indices of nodes in component 2.
#         X (array): Label matrix for nodes.
#         O (array): Binding matrix.
#         A (array): Adjacency matrix.
#         viable_pairs (array): Array to store viable pairs of nodes to connect.
#         possible_pairs (array): Array to store all possible links between the two components.

#     Returns:
#         viable_pairs (array): List of viable pairs of nodes to connect.
#         possible_pairs (array): List of all possible links between the two components.
#         counter (int): Number of viable pairs.
#     """
    
#     counter = 0
#     possible_counter = 0

#     for u in nb.prange(len(comp1)):
#         e1 = comp1[u]
#         for v in nb.prange(len(comp2)):
#             e2 = comp2[v]
            
#             # Determine the labels of e1 and e2
#             label1 = X[e1].argmax()
#             label2 = X[e2].argmax()
            
#             # Count how many neighbors of e1 have label2
#             count1 = np.dot(A,X)[e1,label2]
#             # Count how many neighbors of e2 have label1
#             count2 = np.dot(A,X)[e2,label1]
#             # Check connection viability
#             if count1 < O[label1, label2] and count2 < O[label2, label1]:
#                 viable_pairs[counter] = (e1, e2)
#                 A[e1,e2] = float(1)
#                 A[e2,e1] = float(1)
#                 counter += 1
            
#             # # Add to possible links if O allows connections
#             # if O[label1, label2] > 0 and O[label2, label1] > 0:
#             #     possible_pairs[possible_counter] = (e1,e2)
#             #     possible_counter += 1

#     return viable_pairs, counter

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # if isinstance(key, str):
    #     # Encode the key as ASCII
    #     key = key.encode('ascii', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        value = value.encode('ascii', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG

class NetAssembly:
    """
    Simulate nodes interacting to form a network.
    Each node has a labeled defined by the matrix X, where X_{ij}=1 if node i has particle type j.
    Each particle type has binding rules encapsulated in O, where O_{ij} is the number of possible 
    connections a particle i can have with a particle j.
    Disconnected components in the network interact at a rate dependent on the number of nodes in the component.
    When they interact, the two components try to connect as much as possible.
    At each step, unstable nodes (those not full bound according to O) detach from their connected component at a rate 'detachment_rate'.

    Attributes:
        g (networkx or graph-tool) - underlying network
        X (ndarray) - node labels in array form
        O (ndarray) - binding matrix
        N_dist (ndarray) - node distribution
        labels (ndarray) - node labels
        directed (bool) - True if directed network
        graph_tool - True if using graph_tool
        A (scipy matrix) - adjacency matrix 
    """
    def __init__(self,X,O,N_dist,directed=False,graph_tool=False,system_energy=1):
        self.X = X
        self.O = O
        self.labels = X.argmax(axis=1)
        self.N_dist = N_dist
        self.N = np.sum(N_dist)
        self.graph_tool = graph_tool
        self.directed = directed
        self.system_energy = system_energy

        # Create network
        if graph_tool:
            self.g = gt.Graph(directed=directed)
            self.g.add_vertex(self.N)
            label_property = self.g.new_vertex_property("int")
            self.g.vp.label = label_property
            # Assign node labels
            for i, v in enumerate(self.g.vertices()):
                label_property[v] = self.labels[i]
            # Set fast edge removal
            self.g.set_fast_edge_removal(fast=True)
            self.A = gt.adjacency(self.g,operator=False)
        else:
            self.g = nx.Graph()
            self.g.add_nodes_from(range(self.N), dtype=int)
            self.A = nx.adjacency_matrix(self.g).toarray()

    def update_A(self):
        if self.graph_tool:
            self.A = gt.adjacency(self.g,operator=False).toarray()
        else:
            self.A = nx.adjacency_matrix(self.g).toarray()

    def run(self,T,component=False,link_strength=0,update_A=True,connection_method='probabilistic'):
        """
        Run simulation for T timesteps.

        Parameters:
            T (int) - number of steps
            component (bool) - randomly select connected components rather than nodes
        """
        for t in range(T):
            self._step(component,link_strength,connection_method)
            if update_A:
                self.update_A()

    def _step(self,component,link_strength,connection_method='probabilistic'):
        """
        A single step of simulation.

        Parameters:
            component (bool) - randomly select connected components rather than nodes
        """
        # Select two random nodes
        if component:
            if self.graph_tool:
                self.g.vp.comp, hist = gt.label_components(self.g)
                component_list = np.unique(self.g.vp.comp.a)
                comp1_label, comp2_label = np.random.choice(component_list,size=2,replace=False)
                comp1 = [int(v) for v in self.g.vertices() if self.g.vp.comp[v] == comp1_label]
                comp2 = [int(v) for v in self.g.vertices() if self.g.vp.comp[v] == comp2_label]
            else:
                comp1, comp2 = np.random.choice(list(nx.connected_components(self.g)),size=2,replace=False)
            comp1 = list(comp1)
            comp2 = list(comp2)

        else:
            n1, n2 = np.random.choice(self.N,size=2,replace=False)

            # Get components of the nodes
            if self.graph_tool:
                self.g.vp.comp, hist = gt.label_components(self.g)
                comp1_label = self.g.vp.comp[n1]
                comp2_label = self.g.vp.comp[n2]
                comp1 = [int(v) for v in self.g.vertices() if self.g.vp.comp[v] == comp1_label]
                comp2 = [int(v) for v in self.g.vertices() if self.g.vp.comp[v] == comp2_label]
            else:
                comp1 = list(nx.node_connected_component(self.g, n1))
                comp2 = list(nx.node_connected_component(self.g, n2))
            # Check whether components are the same
            if sorted(comp1) == sorted(comp2):
                return
        
        np.random.shuffle(comp1)
        np.random.shuffle(comp2)

        # Get number of each node type in each component
        # comp1_dist = self.X[comp1].sum(axis=0)
        # comp2_dist = self.X[comp2].sum(axis=0)

        # Check how many possible links between each node type

                
        # Create neighbor dictionary
        neighbors, neighbortypes, offsets = self.build_adjacency_list()
        
        viable_pairs, counter = compute_viable_pairs_and_possible_links(
            comp1, comp2, self.X, self.O, neighbors, offsets, np.zeros((len(comp1)*len(comp2),2),dtype=int),np.zeros((len(comp1)*len(comp2),2),dtype=int))
        # viable_pairs, counter = compute_viable_pairs_and_possible_links(
            # comp1, comp2, self.X.astype(float), self.O.astype(float), copy.deepcopy(self.A.astype(float)), np.zeros((len(comp1)*len(comp2),2),dtype=int))
        # print(counter)
        if connection_method == 'maximally_connect_check':
            if np.allclose(np.sort(viable_pairs),np.sort(possible_pairs)):
                if self.graph_tool:
                    self.g.add_edge_list(list(viable_pairs[:counter]))
                else:
                    self.g.add_edges_from(list(viable_pairs[:counter]))

        elif connection_method == 'probabilistic':
            # Generate random number
            r = np.random.random()
            # Compute probability of connection based on energy
            # p = np.exp(-(len(possible_pairs) - counter)*self.system_energy)
            p = 1 - np.exp(-counter*self.system_energy)
            if r <= p:
                # Check how many neighbors each node can add
                neighbor_matrix = self.A@self.X
                max_neighbor_matrix = self.X@self.O
                neighbor_count = {i: max_neighbor_matrix[self.X[i].argmax()] - neighbor_matrix[i] for i in np.append(comp1,comp2)}
                for pair in viable_pairs[:counter]:
                    print(pair)
                    if neighbor_count[pair[0]][self.X[pair[1]].argmax()] > 0 and neighbor_count[pair[1]][self.X[pair[0]].argmax()] > 0:
                        if self.graph_tool:
                            self.g.add_edge(pair)
                        else:
                            self.g.add_edge(*pair)
                        neighbor_count[pair[0]][self.X[pair[1]].argmax()] -= 1
                        neighbor_count[pair[1]][self.X[pair[0]].argmax()] -= 1
                # if self.graph_tool:
                #     self.g.add_edge_list(list(viable_pairs[:counter]))
                # else:
                #     self.g.add_edges_from(list(viable_pairs[:counter]))

        elif connection_method == 'maximally_connect':
            if self.graph_tool:
                self.g.add_edge_list(list(viable_pairs[:counter]))
            else:
                self.g.add_edges_from(list(viable_pairs[:counter]))
    
        # Batch detachment check
        # Caculate detachment probability
        detach_prob = np.random.random(size=self.N)
        
        # Count disconnected links in network
        link_count = (self.X@self.O - self.A@self.X).sum(axis=1)

        # Select node to detach
        node_to_detach = np.random.choice(list(self.g.nodes()))
        if detach_prob[node_to_detach] < 1 - np.exp(-link_strength*link_count[node_to_detach]):
            if self.graph_tool:
                self.g.remove_edges(self.g.get_out_edges(node_to_detach))
            else:
                self.g.remove_edges_from(list(self.g.edges(node_to_detach)))
        # nodes_to_detach = np.where(detach_prob < np.exp(-link_strength*link_count))[0]
        
        # for node in nodes_to_detach:
        #     if self.graph_tool:
        #         self.g.remove_edges(self.g.get_out_edges(node))
        #     else:
        #         self.g.remove_edges_from(list(self.g.edges(node)))

    def build_adjacency_list(self):
        """Convert a graph's neighbors to adjacency list format."""
        neighbors = []
        offsets = np.zeros(self.N + 1, dtype=np.int32)
    
        # Collect neighbors and offsets
        idx = 0
        for node in range(self.N):
            if self.graph_tool:
                node_neighbors = list(self.g.get_all_neighbors(node))  # Get neighbors for each node
            else:
                node_neighbors = list(self.g.neighbors(node))
            neighbors.extend(node_neighbors)
            idx += len(node_neighbors)
            offsets[node + 1] = idx  # Store cumulative offset
        neighbortypes = [self.labels[neighbors[j]] for j in range(len(neighbors))]
        return np.array(neighbors, dtype=np.int32), np.array(neighbortypes,dtype=np.int32), offsets

    

    def draw(self,with_networkx=True,figsize=(5,5),color_key=None,**kwargs):
        """
        Draw network where nodes are labeled by particle type.
        """
        if with_networkx:
            if self.graph_tool:
                plot_g = nx.from_scipy_sparse_array(self.A)
            else:
                plot_g = self.g

            fig = plt.figure(figsize=figsize)
            
            # Get color for particles
            if color_key is None:
                color_key = plt.cm.rainbow(np.linspace(0, 1, self.O.shape[0]))
            node_colors = [color_key[j] for j in self.X.argmax(axis=1)]

            nx.draw(plot_g,node_color=node_colors,**kwargs)
            plt.show()

                
        else:
            if self.graph_tool:
                plot_g = self.g
            else:
                plot_g = nx2gt(self.g)

            fig = plt.figure(figsize = figsize)

            # Get color for particles
            vcolor = plot_g.new_vp("vector<double>") 
            if color_key is None:
                color_key = plt.cm.rainbow(np.linspace(0, 1, self.O.shape[0]))
            for i in range(self.N):
                vcolor[i] = color_key[self.X[i].argmax()]

            gt.graph_draw(plot_g,vertex_fill_color=vcolor,**kwargs)
            plt.show()

            
    
    
            
    
        