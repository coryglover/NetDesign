import numpy as np
import networkx as nx
import itertools

class NetDesign:
    """
    Simulate nodes interacting to form a network.
    Each node has a labeled defined by the matrix X, where X_{ij}=1 if node i has particle type j.
    Each particle type has binding rules encapsulated in O, where O_{ij} is the number of possible 
    connections a particle i can have with a particle j.
    There are three types of events
        1) Merging: Here two nodes connect as well as their k-hop ego network. 
            -There are |C|(|C|+1)/2 possible events, where |C| is the amount of CCs.
            -The rate depends on how many potential connections there are between the CCs.            
        2) Detachment: Here a node detaches from the graph. 
            -There are |N| such events
            -The rate depends on how strongly the component in question is bounded to the bulk.
        3) The O matrix is updated and coarse-grained. 
            -There is one such event.
            -Rate is system specific.
    The simulation are then based on the Gillespie algorithm where the rate vector has length |C|(|C|+1)/2 + |N| + 1
    
    Attributes:
        g (networkx) - underlying network
        X (ndarray) - node labels in array form
        O (ndarray) - binding matrix
        labels (dictionary) - node ID to label
        N (scalar) - amount of nodes in the network

        potential_links (NxN ndarray) - Amount of potential links between two nodes
        comp_compatibility (CxC ndarray) - Amount of potential links between two components. The components are ordered according to the smallest node ID they contain. Last N-|C| rows/columns are zero.
        type_degrees (M ndarray) - Total possible degree of the node types
        node_degrees (N ndarray) - The actual degree of the nodes
        connected_components (generator for sets of nodes) - connected components in the underlying graph

        rates_attach (C(C+1)/2 ndarray) - merging rates
        rates_detach (N ndarray) - detachment rates
        rates  (N+C(C+1)/2+1 ndarray) - concatenated attach and detach rates
    """
    def __init__(self,X,O,target_network):
        #Defining properties
        self.X = X.astype(int)
        self.O = O.astype(int)
        self.target_network = target_network
        self.N = X.shape[0]
        self.labels = {i:X[i].argmax() for i in range(self.N)}

        #graph
        self.g = nx.Graph()
        self.g.add_nodes_from(np.arange(self.N))
        nx.set_node_attributes(self.g,self.labels,name="labels")
        nx.set_node_attributes(self.g,np.arange(self.N),name='component')
        self.A = nx.adjacency_matrix(self.g).toarray()

        # Create component graph
        self.coarse_grain(1,1)

        #Following the progress
        self.ccs = []
        self.lccs = []
        self.lccs2= []
        self.ts = []
        self.event = []
    
    def run(self,T,k=10,diagonal_merging_factor=1000,kappa_A=None,kappa_D=None,kappa_O=None,relative_rates=None):
        #Initialize
        t = 0
        counter = 0

        if kappa_D != None and kappa_A != None:
            attach_x = np.ones(len(self.connected_components)*(len(self.connected_components) - 1)/2)
            self.rates_attach = kappa_A*attach_x / np.sum(attach_x)
            self.rates_detach = kappa_D*np.zeros(self.N)

        else:
            self.rates_attach = self.comp_compatibility[np.triu_indices(self.C)]
            self.rates_detach = np.zeros(self.N)
        
        self.rates = np.concatenate((self.rates_attach,self.rates_detach,kappa_O))

        while counter < T:
            #Draw two uniform RVs
            u1,u2 = np.random.uniform(0,1,3)
            #Make sure the simulation doesn't run too long
            if np.sum(self.rates)==0:# or counter > 500000 or np.sum(self.rates_detach[list(self.lccs2[-1])]) == 0:
                break

            #Calculate the time step
            tau = -np.log(u1)/np.sum(self.rates)

            #Choose which event will happen
            event_id = np.searchsorted(np.cumsum(self.rates)/np.sum(self.rates),u2)
            if event_id<self.rates_attach.size:
                self.attach_event(event_id,k,diagonal_merging_factor,kappa_A,kappa_D,kappa_O,relative_rates)
            elif event_id == len(self.rates):
                self.coarse_grain(kappa_A,kappa_O)
            else:
                self.detach_event(event_id,k,diagonal_merging_factor,kappa_A,kappa_D,kappa_O,relative_rates)

            #Increase time and amount of steps excecuted
            t += tau
            counter += 1

            #Save properties of the state
            self.event.append(np.sum(self.rates_detach)/np.sum(self.rates))
            self.ccs.append(len(self.connected_components))
            self.ts.append(t)
            self.lccs.append(max([len(c) for c in self.connected_components]))
            self.lccs2.append(np.sort(np.array([len(c) for c in self.connected_components]))[-5:])

            # if np.sum(self.rates_detach[list(self.lccs2[-1])]) == 0:
            #     break
        
    def attach_event(self,event_id,k,diagonal_merging_factor,kappa_A,kappa_D,kappa_O):
        """
        Attachment event. Connect two components based on O matrix and update underlying network.
        """
        # Get quotient nodes to attach
        ix_L = np.triu_indices(self.C)[0][event_id]
        ix_R = np.triu_indices(self.C)[1][event_id]

        # Get ego networks of quotient nodes
        c_L = nx.ego_graph(self.g_comp,ix_L,radius=k).nodes()
        c_R = nx.ego_graph(self.g_comp,ix_R,radius=k).nodes()

        # Randomly order nodes
        np.random.shuffle(c_L)
        np.random.shuffle(c_R)

        # Attempt to connect 
        for i in c_L:
            for j in c_R:
                # Connect components if possible
                self.attach_comps(i,j)
        
        self.update_rates(event_id,True,kappa_A,kappa_D,kappa_O)

    def update_rates(self,event_id,attach,kappa_A,kappa_D,kappa_O):
        if attach:
            ix_L = np.triu_indices(self.C)[0][event_id]
            ix_R = np.triu_indices(self.C)[1][event_id]

            # Get nodes in each component
            for n in self.connected_components[ix_L]:
                x = self.g.degree(n) / self.O[self.X[n].argmax()].sum()
                self.rates_detach[n] = kappa_D * (1-x) * np.heaviside(x)
            for n in self.connected_components[ix_R]:
                x = self.g.degree(n) / self.O[self.X[n].argmax()].sum()
                self.rates_detach[n] = kappa_D*(1-x) * np.heaviside(x)
        
        else:
            self.rates_detach[n] = 0
        
        self.rates = np.concatenate(self.rates_attach,self.rates_detach,kappa_O)

    def detach_event(self,event_id,k,diagonal_merging_factor,kappa_A,kappa_D,kappa_O):
        """
        Detachment event. Node detaches.
        """
        #Find the node ID of the node that is detaching
        ix = event_id - self.N*(self.N+1)//2

        # Remove nodes edges
        for n in nx.neighbors(self.g,ix):
            self.g.remove_edge(ix,n)
        
        self.update_rates(event_id,False,kappa_A,kappa_D,kappa_O)
        

    def attach_comps(self,i,j):
        """
        Attach components i and j according to coarse graining.
        """
        # Choose mapping for each subgraph
        subgraph_i = self.component_to_subgraph[i]
        subgraph_j = self.component_to_subgraph[j]

        mapping_i = np.random.choice(self.subgraph_dict[subgraph_i])
        mapping_j = np.random.choice(self.subgraph_dict[subgraph_j])

        # Get nodes that may connected
        comp_1_avail_nodes = []
        comp_2_avail_nodes = []
        for k, n in mapping_i.items():
            if sorted(list(nx.neighbors(self.target_network,n))) not in list(mapping_i.values()):
                comp_1_avail_nodes.append(k)
        for k, n in mapping_j.items():
            if sorted(list(nx.neighbors(self.target_network,n))) not in list(mappint_j.values()):
                comp_2_avail_nodes.append(k)
        
        comp_1_avail_nodes = np.array(comp_1_avail_nodes)
        comp_2_avail_nodes = np.array(comp_2_avail_nodes)
        np.random.shuffle(comp_1_avail_nodes)
        np.random.shuffle(comp_2_avail_nodes)

        # Add edges
        for n1 in comp_1_avail_nodes:
            for n2 in comp_2_avail_nodes:
                label1 = self.X[n1].argmax()
                label2 = self.X[n2].argmax()
                if self.A@self.X[n1,label2] < self.O[label1,label2] and self.A@self.X[n2,label1] < self.O[label2,label1]:
                    self.g.add_edge(n1,n2)
                    self.g_comp.add_edge(i,j)
                    self.A = nx.adjacency_matrix(self.g).toarray()


    def coarse_grain(self,kappa_A,kappa_O):
        """
        Updates quotient map based on underlying target network.
        """

        def node_match(n1,n2):
            return self.g.nodes[n1]['labels'] == self.g.nodes[n2]['labels']
        
        # Get components
        self.connected_components = [sorted(list(c)) for c in nx.connected_components(self.g)]
        self.connected_components = sorted(self.connected_components, key=lambda c: c[0])

        # Find copies of connected components in original graph
        # Make subgraph list
        self.subgraphs = []
        self.component_to_subgraph = {}
        self.quotient_X = np.zeros((len(self.connected_components),len(self.connected_components)))
        for i, c in enumerate(self.connected_components):
            H = nx.subgraph(self.target_network,c)

            # Check whether subgraph already exists in quotient graph
            found = False
            for other_subgraphs in self.subgraphs:
                if nx.is_isomorphic(H,other_subgraphs,node_match=node_match):
                    self.quotient_X[i,self.subgraphs.index(other_subgraphs)] = 1
                    self.component_to_subgraph[i] = self.subgraphs.index(other_subgraphs)
                    found = True
                if found:
                    break
            if found:
                continue
            else:
                self.subgraphs.append(H)
                self.component_to_subgraph[i] = self.subgraphs.index(H)
        
        # Initilize new O matrix
        self.quotient_X = self.quotient_X[:,len(self.subgraphs)]

        # Set quotient node types
        self.subgraph_dict = {i:None for i in range(len(self.subgraphs))}

        for j, H in enumerate(self.subgraphs):
            GM = nx.isomorphism.GraphMatcher(self.target_network, H,node_match=node_match)
            # Find subgraphs in target network
            self.subgraph_dict[j] = list(GM.subgraph_isomorphisms_iter())
        
        # Update component network
        self.g_comp = nx.Graph()
        self.g_comp.add_nodes_from(np.arange(len(self.connected_components)))

        # Update rates
        attach_rate = np.ones(len(self.connected_components))
        self.rates_attach = kappa_A * attach_rate / np.sum(attach_rate)
        self.rates = np.concatenate(self.rates_attach,self.rates_detach,kappa_O)
        
            

# class network:
#     """
#     Simulate nodes interacting to form a network.
#     Each node has a labeled defined by the matrix X, where X_{ij}=1 if node i has particle type j.
#     Each particle type has binding rules encapsulated in O, where O_{ij} is the number of possible 
#     connections a particle i can have with a particle j.
#     There are two types of events
#         1) Merging: Here two connected components maximally connect. 
#             -There are |C|(|C|+1)/2 possible events, where |C| is the amount of CCs.
#             -The rate depends on how many potential connections there are between the CCs            
#         2) Detachment: Here a node detaches from the graph. 
#             -There are |V| such events
#             -The rate depends on how strongly the node in question is bounded to the bulk. 
#     The simulation are then based on the Gillespie algorithm where the rate vector has length |V|(|V|+1)/2 + |V|
    
#     Attributes:
#         g (networkx) - underlying network
#         X (ndarray) - node labels in array form
#         O (ndarray) - binding matrix
#         labels (dictionary) - node ID to label
#         N (scalar) - amount of nodes in the network

#         potential_links (NxN ndarray) - Amount of potential links between two nodes
#         comp_compatibility (NxN ndarray) - Amount of potential links between to components. The components are ordered according to the smallest node ID they contain. Last N-|C| rows/columns are zero.
#         type_degrees (M ndarray) - Total possible degree of the node types
#         node_degrees (N ndarray) - The actual degree of the nodes
#         connected_components (generator for sets of nodes) - connected components in the graph

#         rates_attach (N(N+1)/2 ndarray) - merging rates
#         rates_detach (N ndarray) - detachment rates
#         rates  (N(N+3)/2 ndarray) - concatenated attach and detach rates



#     """
#     def __init__(self,X,O,target_network):
#         #Defining properties
#         self.X = X.astype(int)
#         self.O = O.astype(int)
#         self.quotient_O = O.astype(int)
#         self.target_network = target_network
#         self.N = X.shape[0]
#         self.labels = {i:X[i].argmax() for i in range(self.N)}

#         #graph
#         self.g = nx.Graph()
#         self.g.add_nodes_from(np.arange(self.N))
#         nx.set_node_attributes(self.g,self.labels,name="labels")
#         nx.set_node_attributes(self.g,np.arange(self.N),name='component')

#         #Component graph. Identical to original graph in first steps
#         self.g_comp = self.g.copy()
#         self.C = self.g_comp.number_of_nodes()

#         #Properties necessary for defining the steps
#         self.potential_links = self.X@self.O@self.X.T
#         self.comp_compatibility = np.heaviside(self.X@self.O@self.X.T,0).astype(int)
#         self.type_degrees = np.sum(O,axis=1)
#         self.node_degrees = np.zeros(self.N)
#         #self.connected_components = sorted(nx.connected_components(self.g), key=len, reverse=True)
#         self.connected_components = [sorted(list(c)) for c in nx.connected_components(self.g)]
#         self.connected_components = sorted(self.connected_components, key=lambda c: c[0])

#         #Following the progress
#         self.ccs = []
#         self.lccs = []
#         self.lccs2= []
#         self.ts = []
#         self.event = []

#     '''
#     This function runs the simulation following the gillespie algorithm.    
#     '''
#     def run(self,T,k=10,diagonal_merging_factor=1000,kappa_A=None,kappa_D=None,kappa_O=None,relative_rates=None):
#         #Initialize
#         t = 0
#         counter = 0

#         if kappa_D != None and kappa_A != None:
#             self.rates_attach = kappa_A*self.comp_compatibility[np.triu_indices(self.C)]
#             self.rates_detach = kappa_D*np.zeros(self.C)

#         else:
#             self.rates_attach = self.comp_compatibility[np.triu_indices(self.C)]
#             self.rates_detach = np.zeros(self.C)
        
#         self.rates = np.concatenate((self.rates_attach,self.rates_detach,kappa_O))

#         while counter < T:
#             #Draw two uniform RVs
#             u1,u2 = np.random.uniform(0,1,3)
#             #Make sure the simulation doesn't run too long
#             if np.sum(self.rates)==0:# or counter > 500000 or np.sum(self.rates_detach[list(self.lccs2[-1])]) == 0:
#                 break

#             #Calculate the time step
#             tau = -np.log(u1)/np.sum(self.rates)

#             #Choose which event will happen
#             event_id = np.searchsorted(np.cumsum(self.rates)/np.sum(self.rates),u2)
#             if event_id<self.rates_attach.size:
#                 self.attach_event(event_id,k,diagonal_merging_factor,kappa_A,kappa_D,kappa_O,relative_rates)
#             elif event_id == len(self.rates):
#                 self.update_O(event_id,k,diagonal_merging_factor,kappa_A,kappa_D,kappa_O,relative_rates)
#             else:
#                 self.detach_event(event_id,k,diagonal_merging_factor,kappa_A,kappa_D,kappa_O,relative_rates)

#             #Increase time and amount of steps excecuted
#             t += tau
#             counter += 1

#             #Save properties of the state
#             self.event.append(np.sum(self.rates_detach)/np.sum(self.rates))
#             self.ccs.append(len(self.connected_components))
#             self.ts.append(t)
#             self.lccs.append(max([len(c) for c in self.connected_components]))
#             self.lccs2.append(np.sort(np.array([len(c) for c in self.connected_components]))[-5:])

#             # if np.sum(self.rates_detach[list(self.lccs2[-1])]) == 0:
#             #     break
            
#     '''
#     This function defines the attach events
#     '''
#     def attach_event(self,event_id,diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates,k):
#         #Find which nodes are merging
#         ix_L = np.triu_indices(self.C)[0][event_id]
#         ix_R = np.triu_indices(self.C)[1][event_id]

#         # Get ego networks of nodes
#         c_L = nx.ego_graph(self.g_comp,ix_L,radius=k)
#         c_R = nx.ego_graph(self.g_comp,ix_R,radius=k)

#         #Find which nodes in each of the two components are connecting
#         coupling_nodes_L = set()
#         coupling_nodes_R = set()
#         for node1 in c_L:
#             for node2 in c_R:
#                 if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
#                     coupling_nodes_L.add(node1)
#                     coupling_nodes_R.add(node2)

#         #Remove the nodes involved in the merging and recalculate the comp_compatibility
#         self.update_comp_compatibility_attach(ix_L,ix_R,coupling_nodes_L,coupling_nodes_R,1)

#         #Delete the column and rows in the com_compatibility matrix related to c_R, and shift the columns and rows such that the zeros are on the right/bottom
#         #Skip this step if we are self-merging
#         if ix_L!=ix_R:
#             self.shift_component_matrix_attach(ix_R)

#         #Update the graph and potential links between nodes
#         for node1 in coupling_nodes_L:
#             for node2 in coupling_nodes_R:
#                 if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
#                     self.g.add_edge(node1,node2)
#                     self.node_degrees[node1]+=1
#                     self.node_degrees[node2]+=1
#                     self.potential_links[node1,np.array(list(self.labels.values()))==self.labels[node2]] -= 1
#                     self.potential_links[node2,np.array(list(self.labels.values()))==self.labels[node1]] -= 1
        
#         #Get the new connected components in the correct order to match those of the comp_compatibility matrix. (This step I think could be made more efficient)
#         self.connected_components = [sorted(list(c)) for c in nx.connected_components(self.g)]
#         self.connected_components = sorted(self.connected_components, key=lambda c: c[0])

#         #Add the nodes involved in the merging and recalculate the comp_compatibility
#         self.update_comp_compatibility_attach(ix_L,ix_R,coupling_nodes_L,coupling_nodes_R,2)

#         #Update the rates based on the comp_compatibility and the node_degrees
#         self.update_rates(np.array(list(coupling_nodes_L|coupling_nodes_R)),diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates)


#     '''
#     This function defines the detach events
#     '''
#     def detach_event(self,event_id,diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates):
#         #Find the node ID of the node that is detaching
#         ix = event_id - self.N*(self.N+1)//2

#         #Find the nodes in that nodes connected component
#         c = nx.node_connected_component(self.g,ix)

#         #Reinitalize the detaching node's degree and reinstate its potential links with all other nodes
#         self.node_degrees[ix] = 0
#         for i in range(self.N):
#             self.potential_links[ix,i] = self.O[self.labels[ix],self.labels[i]]
        
#         #Find the neighbours of the detaching nodes
#         neighbors = list(self.g.neighbors(ix))

#         #Remove the edges connecting the detaching nodes to its neighbours, recompute the potential_links
#         for node in neighbors:
#             self.potential_links[node,np.array(list(self.labels.values()))==self.labels[ix]] += 1
#             self.g.remove_edge(node,ix)
#             self.node_degrees[node] -= 1

#         #If the detachment event created x new connected components, find where these go in the compt_compatibility matrix and shift zero columns and rows there. Return the smallest node IDs of these components.
#         smallest_nodes_new = self.shift_component_matrix_detach(c)

#         #Update the connected components
#         self.connected_components = [sorted(list(c)) for c in nx.connected_components(self.g)]
#         self.connected_components = sorted(self.connected_components, key=lambda c: c[0])

#         #Update the comp_compatibility matrix using the smallest node ID information we just found
#         self.update_comp_compatibility_detach(smallest_nodes_new)

#         #Update the rates based on the comp_compatibility and the node_degrees
#         self.update_rates(np.array(neighbors+[ix]),diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates)
        
        

#     '''
#     This function reorganizes the comp_compatibility matrix after two components merged.
#     '''
#     def shift_component_matrix_attach(self,comp_R):
#         #Delete the row and column associated with the second merging component
#         self.comp_compatibility[comp_R,:] = 0
#         self.comp_compatibility[:,comp_R] = 0
#         #Roll the empty row and column to the bottom and right, respectively
#         self.comp_compatibility[comp_R:,:] = np.roll(self.comp_compatibility[comp_R:,:],-1,axis=0)
#         self.comp_compatibility[:,comp_R:] = np.roll(self.comp_compatibility[:,comp_R:],-1,axis=1)

#     '''
#     This function reorganizes the comp_compatibility matrix after the node detachment created new components
#     '''
#     def shift_component_matrix_detach(self,c_ix):
#         #These are the new components that have formed from the connected component of the detaching node (c_ix). Sort them according to the smallest node_ID they contain.
#         new_components = [sorted(list(c)) for c in nx.connected_components(self.g.subgraph(c_ix))]
#         new_components = sorted(new_components, key=lambda c: c[0])

#         #The set of smallest node IDs in each of the disconnected components. Note that self.connected_components has not been updated yet so this is for the old situation
#         smallest_nodes_before = np.array([c[0] for c in self.connected_components])
        
#         #The set of smallest node IDs in each of the disconnected components created in the detachment even.
#         smallest_nodes_new = np.array([c[0] for c in new_components])
        
#         #Find the ones that are new, those need to be inserted into the matrix
#         to_be_insterted = np.setdiff1d(smallest_nodes_new,smallest_nodes_before)
        
#         #Find the locations that the new components need to inserted into
#         locs = np.searchsorted(smallest_nodes_before,to_be_insterted)

#         #Take empty rows and columns from the sides of the matrix and roll them to the proper location
#         for i,loc in enumerate(locs):
#             self.comp_compatibility[loc+i:len(self.connected_components)+i+1,:] = np.roll(self.comp_compatibility[loc+i:len(self.connected_components)+i+1,:],1,axis=0)
#             self.comp_compatibility[:,loc+i:len(self.connected_components)+i+1] = np.roll(self.comp_compatibility[:,loc+i:len(self.connected_components)+i+1],1,axis=1)

#         return smallest_nodes_new
    
#     '''
#     This function updates the comp_compatibility matrix during an attach event
#     '''
#     def update_comp_compatibility_attach(self,comp_L,comp_R,coupling_nodes_L,coupling_nodes_R,step):
#         #Step 1 indicates that the graph has not been updated yet, we are still in the old situation
#         if step == 1:
#             #Self merging is treated differently
#             if comp_L!=comp_R:
#                 for i in range(len(self.connected_components)):
#                     #We only look at the interactions between the merging components with 'spectator' components. The interactions LR, LL, RL, RR will all be zero after the merging is complete.
#                     if i==comp_L or i == comp_R:
#                         continue
#                     #How would the comp_compatibility be if the nodes actively participating in the merging did not exist?
#                     for node1 in self.connected_components[i]:
#                         for node2 in coupling_nodes_L:
#                             if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
#                                 self.comp_compatibility[i,comp_L] -= 1
#                                 self.comp_compatibility[comp_L,i] -= 1
#                         for node2 in coupling_nodes_R:
#                             if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
#                                 self.comp_compatibility[i,comp_R] -= 1
#                                 self.comp_compatibility[comp_R,i] -= 1
#                     #Prepare the comp_compatibility matrix for the delition of the comp_R row and column
#                     self.comp_compatibility[i,comp_L] += self.comp_compatibility[i,comp_R]
#                     self.comp_compatibility[comp_L,i] += self.comp_compatibility[comp_R,i]
#             #If self merging then comp_L and comp_R contain the same nodes. Thus some steps that were needed above are not relevant.
#             else:
#                 for i in range(len(self.connected_components)):
#                     if i==comp_L:
#                         continue
#                     for node1 in self.connected_components[i]:
#                         for node2 in coupling_nodes_L:
#                             if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
#                                 self.comp_compatibility[i,comp_L] -= 1
#                                 self.comp_compatibility[comp_L,i] -= 1

#         #Step 2 implies that we have already done the merging and updated the network and the potential_links matrix. Here comp_R no longer exists
#         if step == 2:
#             for i in range(len(self.connected_components)):
#                 if i==comp_L:
#                     continue
#                 for node1 in self.connected_components[i]:
#                     for node2 in coupling_nodes_L|coupling_nodes_R:
#                         if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
#                             self.comp_compatibility[i,comp_L] += 1
#                             self.comp_compatibility[comp_L,i] += 1
#             self.comp_compatibility[comp_L,comp_L] = 0

#     '''
#     This function updates the comp_compatibility matrix during a detach event
#     '''
#     def update_comp_compatibility_detach(self,smallest_nodes_new):
#         #Find where the smallest nodes of the components affected by the detachment event are located in the comp_compatibility matrix
#         smallest_nodes = np.array([c[0] for c in self.connected_components])
#         locs = np.nonzero(np.isin(smallest_nodes,smallest_nodes_new))[0]

#         #For all of these locations, update how these components interact with the rest of the components.
#         for loc in locs:
#             for i in range(len(self.connected_components)):
#                 self.comp_compatibility[i,loc] = 0
#                 self.comp_compatibility[loc,i] = 0
#                 for node1 in self.connected_components[i]:
#                     for node2 in self.connected_components[loc]:
#                         if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and not np.any(np.array(list(self.g.neighbors(node1)))==node2):
#                             self.comp_compatibility[i,loc] += 1
#                             if i!=loc:
#                                 self.comp_compatibility[loc,i] += 1
    
#     '''
#     This function updates the rate vector based on the comp_compatibility matrix and the degree vector. 
#     We can toggle between independent attachment and detachment and where the ratio between these events is fixed.
#     '''
#     def update_rates(self, involved_nodes, diagonal_merging_factor, link_attractiveness, link_resilience, relative_rates):
#         #Toggle relative size of the self merging events with respect to the rest of the merging events.
#         X = np.full((self.N,self.N),1)
#         Y = np.diag(np.full(self.N,diagonal_merging_factor))

#         #The attachment rate is given by a flattened comp_compatibility matrix where we take into account the relative size of self merging
#         self.rates_attach = ((X+Y)*self.comp_compatibility)[np.triu_indices(self.N)]

#         #The detachment rate is taken such that its zero when the particle is free as well as when its fully connected. In between it decreases with the amount of occupied links
#         for i in range(involved_nodes.size):
#                 x = self.node_degrees[involved_nodes[i]]/self.type_degrees[self.labels[involved_nodes[i]]]
#                 self.rates_detach[involved_nodes[i]] = (1-x)*np.heaviside(x,0)

#         #If link_attractiveness and link_resilience are given then the program does independent attachment.
#         if link_attractiveness != None and link_resilience != None:
#             self.rates = np.concatenate((link_attractiveness*self.rates_attach,link_resilience*self.rates_detach))

#         #If relative_rates is given then the program keeps the ratio fixed.
#         if relative_rates != None:
#             if np.sum(self.rates_attach)!=0 and np.sum(self.rates_detach)!=0:
#                 self.rates = np.concatenate((relative_rates*self.rates_attach/np.sum(self.rates_attach),self.rates_detach/np.sum(self.rates_detach)))
#             else:
#                 self.rates = np.concatenate((self.rates_attach,self.rates_detach))

#     '''
#     This function updates the O matrix to be between components rather than between nodes.
#     '''
#     def update_O(self):
#         pass

# class NetDesign(network):
#     """
#     A class to perform Network Design.

#     The algorithm is as follows:
#     1. A target network is given with a set of node labels.
#     2. Using random walks, common subgraphs are identified, generating a subgraph distribution.
#     3. A collection of particles is created.
#     4. Particles are joined (with respect to component stabilization).
#     5. At defined intervals, the network is refined based on the subgraph distribution.
#     6. The process is repeated until convergence.

#     Attributes:
#         target_network (networkx.Graph): The target network.
#         X (numpy.ndarray): Node labels.
#         O (numpy.ndarray): Match matrix.
#     """
#     def __init__(self, target_network, X, O):
#         self.target_network = target_network
#         self.X = X.astype(int)
#         self.O = O.astype(int)

#         # Intialize network
#         self.g = nx.Graph()
#         self.g.add_nodes_from(np.arange(self.X.shape[0]),)
#         self.g.set_node_attributes({i:j for i,j in enumerate(self.X.argmax(axis=1))})

#         # Establish ego nets of original network as a function of label
#         self.ego_nets = {}
#         for l in self.X.shape[1]:
#             # Get nodes of type l
#             nodes = np.where(self.X.argmax(axis=1)==l)[0]
#             for k in [2,3,4]:
#                 self.ego_nets[k][l] = []
#                 for n in nodes:
#                     self.ego_nets[k][l].append(nx.ego_graph(self.target_network,n,radius=l))
            
    
#     # def random_walk(self, start_node, num_steps):
#     #     """
#     #     Perform a random walk on the target network.

#     #     Args:
#     #         start_node (int): The starting node for the random walk.
#     #         num_steps (int): The number of steps to take in the random walk.

#     #     Returns:
#     #         list: The sequence of nodes visited during the random walk.
#     #     """
#     #     # Initialize variables
#     #     walk = [start_node]
#     #     # Initialize labels of nodes
#     #     walk_labels = [self.X[start_node].argmax()]
#     #     # Perform the random walk
#     #     for _ in range(num_steps):
#     #         neighbors = list(self.target_network.neighbors(walk[-1]))
#     #         if neighbors:
#     #             walk.append(np.random.choice(neighbors))
#     #             walk_labels.append(self.X[walk[-1]].argmax())

#     #     return walk, walk_labels
    
#     def subgraph_distribution(self, num_walks, walk_lengths):
#         """
#         Generate a subgraph distribution based on random walks.
#         Each subgraph is uniquely identified by its node labels.

#         Args:
#             num_walks (int): The number of random walks to perform for each length.
#             walk_lengths (list): The lengths of each random walk.

#         Returns:
#             list: A list of subgraphs generated from the random walks.
#         """
#         # Initialize dictionary to store subgraphs
#         subgraphs = {}
#         for wl in walk_lengths:
#             # Initialize subgraph list
#             subgraphs[wl] = {}
#             for _ in range(num_walks):
#                 # Generate random walk
#                 start_node = np.random.choice(self.target_network.nodes)
#                 walk, walk_labels = self.random_walk(start_node, wl)
#                 # Get network of node label connections found in random walk
#                 subgraph = self.target_network.subgraph(walk)
#                 # Add subgraph with labels to dictionary
#                 # Check whether isomorphic subgraph is already in dictionary
#                 added = False
#                 for key in subgraphs[wl].keys():
#                     if nx.is_isomorphic(subgraph,key[0]) and sorted(walk_labels) == key[1]:
#                         subgraphs[wl][key] += 1
#                         added = True
#                         break

#                 if not added:
#                     subgraphs[wl][(subgraph,sorted(walk_labels))] = 1

#         return subgraphs
    
    
    
    
