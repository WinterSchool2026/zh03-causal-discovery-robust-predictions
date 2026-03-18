import numpy as np

def get_parents(cg, node_idx):
    """Returns nodes that have a directed edge into node_idx (j -> i)"""
    if isinstance(cg, dict):
        # GES format
        adj = cg['G'].graph
    else:
        # PC format
        adj = cg.G.graph
    num_nodes = adj.shape[0]
    # Parent j must have an arrowhead at node_idx (1) and a tail at j (-1)
    return {j for j in range(num_nodes) 
            if adj[node_idx, j] == 1 and adj[j, node_idx] == -1}

def get_children(cg, node_idx):
    """Returns nodes that node_idx points to (i -> j)"""
    if isinstance(cg, dict):
        # GES format
        adj = cg['G'].graph
    else:
        # PC format
        adj = cg.G.graph
    num_nodes = adj.shape[0]
    # Child j must have an arrowhead at j (1) and a tail at node_idx (-1)
    return {j for j in range(num_nodes) 
            if adj[j, node_idx] == 1 and adj[node_idx, j] == -1}

def get_undirected_neighbors(cg, node_idx):
    """Returns nodes connected by an undirected edge (i - j)"""
    if isinstance(cg, dict):
        # GES format
        adj = cg['G'].graph
    else:
        # PC format
        adj = cg.G.graph
    num_nodes = adj.shape[0]
    # Both sides must have tails (-1)
    return {j for j in range(num_nodes) 
            if adj[node_idx, j] == -1 and adj[j, node_idx] == -1}

def get_spouses(cg, node_idx):
    """Returns nodes that share a child with node_idx (i -> child <- spouse)"""
    children = get_children(cg, node_idx)
    # Undirected edges can technically be children in some DAG extensions
    undirected = get_undirected_neighbors(cg, node_idx)
    potential_children = children | undirected
    
    spouses = set()
    for child in potential_children:
        # A spouse is any parent of my child, excluding myself
        child_parents = get_parents(cg, child)
        # Also include undirected neighbors of the child as potential spouses
        child_undirected = get_undirected_neighbors(cg, child)
        
        for p in (child_parents | child_undirected):
            if p != node_idx:
                spouses.add(p)
    return spouses

def get_mb(cg, node_idx):
    """Builds the Markov Blanket from parents, children, spouses, and undirected neighbors"""
    parents = get_parents(cg, node_idx)
    children = get_children(cg, node_idx)
    undirected = get_undirected_neighbors(cg, node_idx)
    spouses = get_spouses(cg, node_idx)
    
    # In a CPDAG, the Markov Blanket is the union of all these
    return parents | children | undirected | spouses