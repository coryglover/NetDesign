import graph_tool.all as gt

def nonlinear_pa(N, m, alpha=1):
    """
    Generate a network with nonlinear preferential attachment using graph_tool.

    Parameters:
        N (int) - number of nodes
        m (int) - number of links attached to incoming nodes
        alpha (float) - preferential attachment exponent

    Returns:
        graph_tool.Graph
    """
    # Generate initial network
    g = gt.Graph(directed=False)
    g.add_edge_list([(0, 1), (1, 2), (2, 0)])

    for i in range(3, N):
        # Get degree sequence
        deg_seq = np.array([v.out_degree() for v in g.vertices()])
        # Get probabilities
        deg_alpha = np.power(deg_seq, alpha)
        prob = deg_alpha / np.sum(deg_alpha)
        
        # Add new node
        v = g.add_vertex()

        # Add links
        # Choose nodes
        links_to_add = np.random.choice(np.arange(i), p=prob, replace=False, size=m)
        # Add links
        for j in links_to_add:
            g.add_edge(v, g.vertex(j))

    # Shuffle node order
    node_order = np.arange(N)
    np.random.shuffle(node_order)
    g = gt.GraphView(g, vfilt=lambda v: node_order[int(v)] < N)
    g = gt.Graph(g, prune=True)

    return g