import numpy as np
import networkx as nx





class network:
    """
    Simulate nodes interacting to form a network.
    Each node has a labeled defined by the matrix X, where X_{ij}=1 if node i has particle type j.
    Each particle type has binding rules encapsulated in O, where O_{ij} is the number of possible 
    connections a particle i can have with a particle j.
    There are two types of events
        1) Merging: Here two connected components maximally connect. 
            -There are |C|(|C|+1)/2 possible events, where |C| is the amount of CCs.
            -The rate depends on how many potential connections there are between the CCs            
        2) Detachment: Here a node detaches from the graph. 
            -There are |V| such events
            -The rate depends on how strongly the node in question is bounded to the bulk. 
    The simulation are then based on the Gillespie algorithm where the rate vector has length |V|(|V|+1)/2 + |V|
    
    Attributes:
        g (networkx) - underlying network
        X (ndarray) - node labels in array form
        O (ndarray) - binding matrix
        labels (dictionary) - node ID to label
        N (scalar) - amount of nodes in the network

        potential_links (NxN ndarray) - Amount of potential links between two nodes
        comp_compatibility (NxN ndarray) - Amount of potential links between to components. The components are ordered according to the smallest node ID they contain. Last N-|C| rows/columns are zero.
        type_degrees (M ndarray) - Total possible degree of the node types
        node_degrees (N ndarray) - The actual degree of the nodes
        connected_components (generator for sets of nodes) - connected components in the graph

        rates_attach (N(N+1)/2 ndarray) - merging rates
        rates_detach (N ndarray) - detachment rates
        rates  (N(N+3)/2 ndarray) - concatenated attach and detach rates



    """
    def __init__(self,X,O):
        #Defining properties
        self.X = X.astype(int)
        self.O = O.astype(int)
        self.N = X.shape[0]
        self.labels = {i:X[i].argmax() for i in range(self.N)}

        #graph
        self.g = nx.Graph()
        self.g.add_nodes_from(np.arange(self.N))
        nx.set_node_attributes(self.g,self.labels,name="labels")

        #Properties necessary for defining the steps
        self.potential_links = self.X@self.O@self.X.T
        self.comp_compatibility = np.heaviside(self.X@self.O@self.X.T,0).astype(int)
        self.type_degrees = np.sum(O,axis=1)
        self.node_degrees = np.zeros(self.N)
        #self.connected_components = sorted(nx.connected_components(self.g), key=len, reverse=True)
        self.connected_components = [sorted(list(c)) for c in nx.connected_components(self.g)]
        self.connected_components = sorted(self.connected_components, key=lambda c: c[0])

        #Following the progress
        self.ccs = []
        self.lccs = []
        self.lccs2= []
        self.ts = []
        self.event = []

    '''
    This function runs the simulation following the gillespie algorithm.    
    '''
    def run(self,T,diagonal_merging_factor=1000,link_attractiveness=None,link_resilience=None,relative_rates=None):
        #Initialize
        t = 0
        counter = 0

        if link_resilience != None and link_resilience != None:
            self.rates_attach = link_attractiveness*self.comp_compatibility[np.triu_indices(self.N)]
            self.rates_detach = link_resilience*np.zeros(self.N)

        else:
            self.rates_attach = self.comp_compatibility[np.triu_indices(self.N)]
            self.rates_detach = np.zeros(self.N)
        
        self.rates = np.concatenate((self.rates_attach,self.rates_detach))

        while counter < T:
            #Draw two uniform RVs
            u1,u2 = np.random.uniform(0,1,2)
            #Make sure the simulation doesn't run too long
            if np.sum(self.rates)==0:# or counter > 500000 or np.sum(self.rates_detach[list(self.lccs2[-1])]) == 0:
                break

            #Calculate the time step
            tau = -np.log(u1)/np.sum(self.rates)

            #Choose which event will happen
            event_id = np.searchsorted(np.cumsum(self.rates)/np.sum(self.rates),u2)
            if event_id<self.rates_attach.size:
                self.attach_event(event_id,diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates)
            else:
                self.detach_event(event_id,diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates)

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
            
    '''
    This function defines the attach events
    '''
    def attach_event(self,event_id,diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates):
        #Find which connected components are merging
        ix_L = np.triu_indices(self.N)[0][event_id]
        ix_R = np.triu_indices(self.N)[1][event_id]
        c_L = self.connected_components[ix_L]
        c_R = self.connected_components[ix_R]

        #Find which nodes in each of the two components are connecting
        coupling_nodes_L = set()
        coupling_nodes_R = set()
        for node1 in c_L:
            for node2 in c_R:
                if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
                    coupling_nodes_L.add(node1)
                    coupling_nodes_R.add(node2)

        #Remove the nodes involved in the merging and recalculate the comp_compatibility
        self.update_comp_compatibility_attach(ix_L,ix_R,coupling_nodes_L,coupling_nodes_R,1)

        #Delete the column and rows in the com_compatibility matrix related to c_R, and shift the columns and rows such that the zeros are on the right/bottom
        #Skip this step if we are self-merging
        if ix_L!=ix_R:
            self.shift_component_matrix_attach(ix_R)

        #Update the graph and potential links between nodes
        for node1 in coupling_nodes_L:
            for node2 in coupling_nodes_R:
                if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
                    self.g.add_edge(node1,node2)
                    self.node_degrees[node1]+=1
                    self.node_degrees[node2]+=1
                    self.potential_links[node1,np.array(list(self.labels.values()))==self.labels[node2]] -= 1
                    self.potential_links[node2,np.array(list(self.labels.values()))==self.labels[node1]] -= 1
        
        #Get the new connected components in the correct order to match those of the comp_compatibility matrix. (This step I think could be made more efficient)
        self.connected_components = [sorted(list(c)) for c in nx.connected_components(self.g)]
        self.connected_components = sorted(self.connected_components, key=lambda c: c[0])

        #Add the nodes involved in the merging and recalculate the comp_compatibility
        self.update_comp_compatibility_attach(ix_L,ix_R,coupling_nodes_L,coupling_nodes_R,2)

        #Update the rates based on the comp_compatibility and the node_degrees
        self.update_rates(np.array(list(coupling_nodes_L|coupling_nodes_R)),diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates)


    '''
    This function defines the detach events
    '''
    def detach_event(self,event_id,diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates):
        #Find the node ID of the node that is detaching
        ix = event_id - self.N*(self.N+1)//2

        #Find the nodes in that nodes connected component
        c = nx.node_connected_component(self.g,ix)

        #Reinitalize the detaching node's degree and reinstate its potential links with all other nodes
        self.node_degrees[ix] = 0
        for i in range(self.N):
            self.potential_links[ix,i] = self.O[self.labels[ix],self.labels[i]]
        
        #Find the neighbours of the detaching nodes
        neighbors = list(self.g.neighbors(ix))

        #Remove the edges connecting the detaching nodes to its neighbours, recompute the potential_links
        for node in neighbors:
            self.potential_links[node,np.array(list(self.labels.values()))==self.labels[ix]] += 1
            self.g.remove_edge(node,ix)
            self.node_degrees[node] -= 1

        #If the detachment event created x new connected components, find where these go in the compt_compatibility matrix and shift zero columns and rows there. Return the smallest node IDs of these components.
        smallest_nodes_new = self.shift_component_matrix_detach(c)

        #Update the connected components
        self.connected_components = [sorted(list(c)) for c in nx.connected_components(self.g)]
        self.connected_components = sorted(self.connected_components, key=lambda c: c[0])

        #Update the comp_compatibility matrix using the smallest node ID information we just found
        self.update_comp_compatibility_detach(smallest_nodes_new)

        #Update the rates based on the comp_compatibility and the node_degrees
        self.update_rates(np.array(neighbors+[ix]),diagonal_merging_factor,link_attractiveness,link_resilience,relative_rates)
        
        

    '''
    This function reorganizes the comp_compatibility matrix after two components merged.
    '''
    def shift_component_matrix_attach(self,comp_R):
        #Delete the row and column associated with the second merging component
        self.comp_compatibility[comp_R,:] = 0
        self.comp_compatibility[:,comp_R] = 0
        #Roll the empty row and column to the bottom and right, respectively
        self.comp_compatibility[comp_R:,:] = np.roll(self.comp_compatibility[comp_R:,:],-1,axis=0)
        self.comp_compatibility[:,comp_R:] = np.roll(self.comp_compatibility[:,comp_R:],-1,axis=1)

    '''
    This function reorganizes the comp_compatibility matrix after the node detachment created new components
    '''
    def shift_component_matrix_detach(self,c_ix):
        #These are the new components that have formed from the connected component of the detaching node (c_ix). Sort them according to the smallest node_ID they contain.
        new_components = [sorted(list(c)) for c in nx.connected_components(self.g.subgraph(c_ix))]
        new_components = sorted(new_components, key=lambda c: c[0])

        #The set of smallest node IDs in each of the disconnected components. Note that self.connected_components has not been updated yet so this is for the old situation
        smallest_nodes_before = np.array([c[0] for c in self.connected_components])
        
        #The set of smallest node IDs in each of the disconnected components created in the detachment even.
        smallest_nodes_new = np.array([c[0] for c in new_components])
        
        #Find the ones that are new, those need to be inserted into the matrix
        to_be_insterted = np.setdiff1d(smallest_nodes_new,smallest_nodes_before)
        
        #Find the locations that the new components need to inserted into
        locs = np.searchsorted(smallest_nodes_before,to_be_insterted)

        #Take empty rows and columns from the sides of the matrix and roll them to the proper location
        for i,loc in enumerate(locs):
            self.comp_compatibility[loc+i:len(self.connected_components)+i+1,:] = np.roll(self.comp_compatibility[loc+i:len(self.connected_components)+i+1,:],1,axis=0)
            self.comp_compatibility[:,loc+i:len(self.connected_components)+i+1] = np.roll(self.comp_compatibility[:,loc+i:len(self.connected_components)+i+1],1,axis=1)

        return smallest_nodes_new
    
    '''
    This function updates the comp_compatibility matrix during an attach event
    '''
    def update_comp_compatibility_attach(self,comp_L,comp_R,coupling_nodes_L,coupling_nodes_R,step):
        #Step 1 indicates that the graph has not been updated yet, we are still in the old situation
        if step == 1:
            #Self merging is treated differently
            if comp_L!=comp_R:
                for i in range(len(self.connected_components)):
                    #We only look at the interactions between the merging components with 'spectator' components. The interactions LR, LL, RL, RR will all be zero after the merging is complete.
                    if i==comp_L or i == comp_R:
                        continue
                    #How would the comp_compatibility be if the nodes actively participating in the merging did not exist?
                    for node1 in self.connected_components[i]:
                        for node2 in coupling_nodes_L:
                            if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
                                self.comp_compatibility[i,comp_L] -= 1
                                self.comp_compatibility[comp_L,i] -= 1
                        for node2 in coupling_nodes_R:
                            if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
                                self.comp_compatibility[i,comp_R] -= 1
                                self.comp_compatibility[comp_R,i] -= 1
                    #Prepare the comp_compatibility matrix for the delition of the comp_R row and column
                    self.comp_compatibility[i,comp_L] += self.comp_compatibility[i,comp_R]
                    self.comp_compatibility[comp_L,i] += self.comp_compatibility[comp_R,i]
            #If self merging then comp_L and comp_R contain the same nodes. Thus some steps that were needed above are not relevant.
            else:
                for i in range(len(self.connected_components)):
                    if i==comp_L:
                        continue
                    for node1 in self.connected_components[i]:
                        for node2 in coupling_nodes_L:
                            if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
                                self.comp_compatibility[i,comp_L] -= 1
                                self.comp_compatibility[comp_L,i] -= 1

        #Step 2 implies that we have already done the merging and updated the network and the potential_links matrix. Here comp_R no longer exists
        if step == 2:
            for i in range(len(self.connected_components)):
                if i==comp_L:
                    continue
                for node1 in self.connected_components[i]:
                    for node2 in coupling_nodes_L|coupling_nodes_R:
                        if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and node1 != node2:
                            self.comp_compatibility[i,comp_L] += 1
                            self.comp_compatibility[comp_L,i] += 1
            self.comp_compatibility[comp_L,comp_L] = 0

    '''
    This function updates the comp_compatibility matrix during a detach event
    '''
    def update_comp_compatibility_detach(self,smallest_nodes_new):
        #Find where the smallest nodes of the components affected by the detachment event are located in the comp_compatibility matrix
        smallest_nodes = np.array([c[0] for c in self.connected_components])
        locs = np.nonzero(np.isin(smallest_nodes,smallest_nodes_new))[0]

        #For all of these locations, update how these components interact with the rest of the components.
        for loc in locs:
            for i in range(len(self.connected_components)):
                self.comp_compatibility[i,loc] = 0
                self.comp_compatibility[loc,i] = 0
                for node1 in self.connected_components[i]:
                    for node2 in self.connected_components[loc]:
                        if self.potential_links[node1,node2]>0 and self.potential_links[node2,node1]>0 and not np.any(np.array(list(self.g.neighbors(node1)))==node2):
                            self.comp_compatibility[i,loc] += 1
                            if i!=loc:
                                self.comp_compatibility[loc,i] += 1
    
    '''
    This function updates the rate vector based on the comp_compatibility matrix and the degree vector. 
    We can toggle between independent attachment and detachment and where the ratio between these events is fixed.
    '''
    def update_rates(self, involved_nodes, diagonal_merging_factor, link_attractiveness, link_resilience, relative_rates):
        #Toggle relative size of the self merging events with respect to the rest of the merging events.
        X = np.full((self.N,self.N),1)
        Y = np.diag(np.full(self.N,diagonal_merging_factor))

        #The attachment rate is given by a flattened comp_compatibility matrix where we take into account the relative size of self merging
        self.rates_attach = ((X+Y)*self.comp_compatibility)[np.triu_indices(self.N)]

        #The detachment rate is taken such that its zero when the particle is free as well as when its fully connected. In between it decreases with the amount of occupied links
        for i in range(involved_nodes.size):
                x = self.node_degrees[involved_nodes[i]]/self.type_degrees[self.labels[involved_nodes[i]]]
                self.rates_detach[involved_nodes[i]] = (1-x)*np.heaviside(x,0)

        #If link_attractiveness and link_resilience are given then the program does independent attachment.
        if link_attractiveness != None and link_resilience != None:
            self.rates = np.concatenate((link_attractiveness*self.rates_attach,link_resilience*self.rates_detach))

        #If relative_rates is given then the program keeps the ratio fixed.
        if relative_rates != None:
            if np.sum(self.rates_attach)!=0 and np.sum(self.rates_detach)!=0:
                self.rates = np.concatenate((relative_rates*self.rates_attach/np.sum(self.rates_attach),self.rates_detach/np.sum(self.rates_detach)))
            else:
                self.rates = np.concatenate((self.rates_attach,self.rates_detach))
                   



                        







    

    
        


