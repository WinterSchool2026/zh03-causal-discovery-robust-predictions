import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random

class SCMGenerator:
    def __init__(self, d):
        self.d = d
        self.n_nodes = d + 1
        self.Y_idx = d 
        self.interventions = {} 
        self.is_fitted = False

    def fit(self, n_parents, n_childs, n_spouses, sparsity, is_linear=True, noise_type='gaussian'):
        self.is_linear = is_linear
        self.noise_type = noise_type
        
        if n_parents + n_childs + n_spouses > self.d:
            raise ValueError("Requested structure exceeds available X nodes.")
            
        indices = np.random.permutation(self.d)
        self.parents_idx = indices[:n_parents]
        self.children_idx = indices[n_parents:n_parents+n_childs]
        self.spouses_idx = indices[n_parents+n_childs:n_parents+n_childs+n_spouses]
        self.others_idx = indices[n_parents+n_childs+n_spouses:]

        self.A = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        
        # Core Y Structure
        for p in self.parents_idx: self.A[p, self.Y_idx] = 1
        for c in self.children_idx: self.A[self.Y_idx, c] = 1
        for s in self.spouses_idx:
            c_target = np.random.choice(self.children_idx)
            self.A[s, c_target] = 1

        self.topo_order = (list(self.parents_idx) + list(self.spouses_idx) + 
                           list(self.others_idx) + [self.Y_idx] + list(self.children_idx))
        self.rank = {node: i for i, node in enumerate(self.topo_order)}

        # Fill Density (Excluding Y nodes to preserve neighborhood constraints)
        possible_edges = [
            (u, v) for u in range(self.n_nodes) for v in range(self.n_nodes)
            if self.rank[u] < self.rank[v] and self.A[u, v] == 0 and u != self.Y_idx and v != self.Y_idx
        ]
        np.random.shuffle(possible_edges)
        
        total_possible = (self.n_nodes * (self.n_nodes - 1)) // 2
        target_count = int(sparsity * total_possible)
        edges_to_add = max(0, target_count - int(np.sum(self.A)))
        
        for i in range(edges_to_add):
            if i < len(possible_edges):
                u, v = possible_edges[i]
                self.A[u, v] = 1

        self.W = self.A * np.random.uniform(1.0, 1.1, size=(self.n_nodes, self.n_nodes))
        self.W *= np.random.choice([-1, 1], size=(self.n_nodes, self.n_nodes))
        self.is_fitted = True
        
    def fit_from_adjacency(self, A, Y_idx, is_linear=True, noise_type='gaussian', weight_low=4.0, weight_high=6.0):
        if A.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(f"Adjacency matrix must be of shape {(self.n_nodes, self.n_nodes)}")

        self.A = A.copy()
        self.Y_idx = Y_idx
        self.is_linear = is_linear
        self.noise_type = noise_type

        # Identify parents, children, spouses of Y
        self.parents_idx = np.where(self.A[:, Y_idx] != 0)[0]
        self.children_idx = np.where(self.A[Y_idx, :] != 0)[0]

        # Spouses: nodes that share a child with Y (excluding Y itself)
        spouse_candidates = set()
        for child in self.children_idx:
            parents_of_child = set(np.where(self.A[:, child] != 0)[0])
            spouse_candidates.update(parents_of_child)
        spouse_candidates.discard(Y_idx)  # Remove Y itself
        spouse_candidates.difference_update(self.children_idx)  # Remove children
        self.spouses_idx = np.array(list(spouse_candidates))

        # Others: nodes that are not parents, children, spouses, or Y
        self.others_idx = np.array(
            [i for i in range(self.n_nodes)
            if i not in self.parents_idx and i not in self.children_idx
            and i not in self.spouses_idx and i != Y_idx]
        )

        # Compute topological order (assuming DAG)
        in_deg = np.sum(self.A, axis=0)
        remaining = set(range(self.n_nodes))
        topo_order = []

        while remaining:
            zero_in_deg = [i for i in remaining if in_deg[i] == 0]
            if not zero_in_deg:
                raise ValueError("Adjacency matrix contains cycles.")
            for node in zero_in_deg:
                topo_order.append(node)
                remaining.remove(node)
                children = np.where(self.A[node, :] != 0)[0]
                for child in children:
                    in_deg[child] -= 1

        self.topo_order = topo_order
        self.rank = {node: i for i, node in enumerate(self.topo_order)}

        # Generate random weights for existing edges
        self.W = self.A * np.random.uniform(weight_low, weight_high, size=(self.n_nodes, self.n_nodes))
        self.W *= np.random.choice([-1, 1], size=(self.n_nodes, self.n_nodes))

        self.is_fitted = True

    def intervention(self, indices, values):
        """Creates a mutilated SCM where specified nodes are forced to specific values."""
        if not self.is_fitted:
            raise RuntimeError("Fit the SCM before intervening.")
        new_scm = copy.deepcopy(self)
        new_scm.interventions = dict(zip(indices, values))
        for node in indices:
            new_scm.A[:, node] = 0 
            new_scm.W[:, node] = 0
        return new_scm

    def _get_noise(self, n_samples):
        if self.noise_type == 'gaussian': return np.random.normal(0, 1, n_samples)
        elif self.noise_type == 'uniform': return np.random.uniform(-1, 1, n_samples)
        elif self.noise_type == 'exponential': return np.random.exponential(1, n_samples)
        elif self.noise_type == 'laplace': return np.random.laplace(0, 1, n_samples)
        return np.random.normal(0, 1, n_samples)

    def sample(self, n_samples):
        """Samples data and returns a clean Pandas DataFrame."""
        data = np.zeros((n_samples, self.n_nodes))
        for node in self.topo_order:
            if node in self.interventions:
                data[:, node] = self.interventions[node]
            else:
                parents = np.where(self.A[:, node] == 1)[0]
                noise = self._get_noise(n_samples)
                if len(parents) == 0:
                    data[:, node] = noise
                else:
                    val = data[:, parents] @ self.W[parents, node]
                    if not self.is_linear: val = np.sin(val)
                    data[:, node] = val + (0.4 * noise)
        
        # Create Column Names: X1, X2... Xd, Y
        cols = [f"X{i}" for i in range(self.d)] + ["Y"]
        return pd.DataFrame(data, columns=cols)


import numpy as np
import random

def generate_dag(num_vars=10, n_parents=2, n_children=2, n_spouses=1, sparsity=0.3, seed=None):
    """
    Generates a DAG adjacency matrix with controlled parents, children, spouses for the target.
    
    Parameters:
    - num_vars: total number of variables (last variable is the target)
    - n_parents: number of parents of the target
    - n_children: number of children of the target
    - n_spouses: number of spouses (nodes sharing children with target)
    - sparsity: probability of random edges among other nodes
    - seed: random seed for reproducibility
    
    Returns:
    - adj_matrix: num_vars x num_vars adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    adj_matrix = np.zeros((num_vars, num_vars), dtype=int)
    target_idx = num_vars - 1
    num_non_target = num_vars - 1
    
    # Ensure the requested parents/children don't exceed available nodes
    if n_parents + n_children > num_non_target:
        raise ValueError("n_parents + n_children cannot exceed the number of available non-target nodes.")
    
    # --- Step 1: Random edges among non-target nodes ---
    # Flowing strictly from smaller to larger indices ensures a topological base
    for i in range(num_non_target):
        for j in range(i + 1, num_non_target):
            if random.random() < sparsity:
                adj_matrix[i, j] = 1
    
    # Determine a split point to enforce: max(parents) < min(children)
    split_min = n_parents
    split_max = num_non_target - n_children
    split_idx = random.randint(split_min, split_max) if split_max >= split_min else split_min
    
    # --- Step 2: Parents of target ---
    if n_parents > 0:
        # Parents strictly drawn from nodes before the split
        possible_parents = list(range(split_idx))
        parents = random.sample(possible_parents, n_parents)
        for p in parents:
            adj_matrix[p, target_idx] = 1
    else:
        parents = []
    
    # --- Step 3: Children of target ---
    if n_children > 0:
        # Children strictly drawn from nodes at or after the split
        possible_children = list(range(split_idx, num_non_target))
        children = random.sample(possible_children, n_children)
        for c in children:
            adj_matrix[target_idx, c] = 1
    else:
        children = []
    
    # --- Step 4: Spouses (nodes sharing children with target) ---
    used_nodes = set(parents + children)
    possible_spouses = [n for n in range(num_non_target) if n not in used_nodes]
    
    if n_spouses > 0 and children:
        max_child = max(children)
        # To avoid cycles, a spouse must have a lower index than the child it connects to
        valid_spouse_candidates = [s for s in possible_spouses if s < max_child]
        
        actual_n_spouses = min(n_spouses, len(valid_spouse_candidates))
        if actual_n_spouses > 0:
            spouse_nodes = random.sample(valid_spouse_candidates, actual_n_spouses)
            for s in spouse_nodes:
                # Filter to only connect the spouse to children that logically appear after it
                valid_children_for_s = [c for c in children if c > s]
                chosen_child = random.choice(valid_children_for_s)
                adj_matrix[s, chosen_child] = 1
                
    return adj_matrix

def plot_cl_graph(cg, data):
    """
    Plots a causal-learn graph using NetworkX with specific color coding 
    for the Markov Blanket of the target variable 'Y'.
    """
    # 1. Handle the format difference between PC and GES
    if isinstance(cg, dict):
        # GES format
        graph_obj = cg['G']
    else:
        # PC format
        graph_obj = cg.G
    # 1. Extract Adjacency Matrix and Column Names
    # 1 = Arrow head, -1 = Tail. We treat 1 as a directed edge.
    adj = graph_obj.graph
    column_names = data.columns.tolist()
    n_nodes = len(column_names)
    
    # NetworkX expects (Source -> Target), so we transpose the '1's
    A = np.where(adj == 1, 1, 0).T 
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    # 2. Identify the Target and its Neighborhood
    if 'Y' not in column_names:
        raise ValueError("Target variable 'Y' not found in the dataframe columns.")
        
    Y_idx = column_names.index('Y')
    
    parents = list(G.predecessors(Y_idx))
    children = list(G.successors(Y_idx))
    spouses = []
    for child in children:
        # Spouses are other parents of your children
        spouses.extend([p for p in G.predecessors(child) if p != Y_idx])
    spouses = list(set(spouses))

    # 3. Apply the Color Map
    color_map = []
    for i in range(n_nodes):
        if i == Y_idx:
            color_map.append('#FFD700')    # Target (Gold)
        elif i in parents:
            color_map.append('#87CEEB')    # Parents (Sky Blue)
        elif i in children:
            color_map.append('#90EE90')    # Children (Light Green)
        elif i in spouses:
            color_map.append('#FA8072')    # Spouses (Salmon)
        else:
            color_map.append('#D3D3D3')    # Others (Grey)

    # 4. Drawing the Graph
    plt.figure(figsize=(12, 9))
    pos = nx.circular_layout(G)
    labels = {i: name for i, name in enumerate(column_names)}

    nx.draw(
        G, pos, labels=labels, with_labels=True, 
        node_color=color_map, node_size=2200, 
        font_size=9, font_weight='bold', 
        edge_color='#444444', width=1.5,
        arrowstyle='-|>', arrowsize=25, 
        connectionstyle='arc3,rad=0.15',
        min_source_margin=15, min_target_margin=15
    )

    plt.title("Causal Discovery: Markov Blanket of Y", fontsize=14, pad=20)
    plt.axis('off')
    plt.show()



def plot_graphs_from_adj(A_list, Y_idx_list=None, column_names_list=None, plot_titles=None):

    if isinstance(A_list, np.ndarray):
        A_list = [A_list]
    n_graphs = len(A_list)

    if Y_idx_list is None:
        Y_idx_list = [A.shape[0]-1 for A in A_list]
    elif isinstance(Y_idx_list, int):
        Y_idx_list = [Y_idx_list] * n_graphs

    if column_names_list is None:
        column_names_list = [None] * n_graphs

    if plot_titles is None:
        plot_titles = [f"Graph {i+1}" for i in range(n_graphs)]

    fig, axes = plt.subplots(1, n_graphs, figsize=(7 * n_graphs, 7))
    if n_graphs == 1:
        axes = [axes]

    for idx, (A, Y_idx, col_names, ax, title) in enumerate(zip(A_list, Y_idx_list, column_names_list, axes, plot_titles)):
        n_nodes = A.shape[0]

        if col_names is None:
            col_names = [f"X{i}" for i in range(n_nodes)]
        col_names[Y_idx] = "Y"

        # Prepare edge lists
        undirected_edges = []
        directed_edges = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if A[i,j] == 1 and A[j,i] == 1:
                    undirected_edges.append((i,j))
                elif A[i,j] == 1:
                    directed_edges.append((i,j))
                elif A[j,i] == 1:
                    directed_edges.append((j,i))

        # Identify Markov Blanket
        parents = [i for i,j in directed_edges if j == Y_idx]
        children = [j for i,j in directed_edges if i == Y_idx]
        spouses = []
        for child in children:
            spouses.extend([i for i,j in directed_edges if j == child and i != Y_idx])
        spouses = list(set(spouses))

        # Color nodes
        color_map = []
        for i in range(n_nodes):
            if i == Y_idx:
                color_map.append('#FFD700')  # Y
            elif i in parents:
                color_map.append('#87CEEB')  # parents
            elif i in children:
                color_map.append('#90EE90')  # children
            elif i in spouses:
                color_map.append('#FA8072')  # spouses
            else:
                color_map.append('#D3D3D3')  # others

        # Layout: larger radius to avoid clipping
        pos = nx.circular_layout(range(n_nodes), scale=1.5)
        labels = {i: col_names[i] for i in range(n_nodes)}

        # Draw nodes and labels
        nx.draw_networkx_nodes(range(n_nodes), pos, node_color=color_map, node_size=2200, ax=ax)
        nx.draw_networkx_labels(range(n_nodes), pos, labels=labels, font_size=10, font_weight='bold', ax=ax)

        # Common arrow properties
        arrow_props_base = dict(lw=1.5, color='#444444', shrinkA=15, shrinkB=15)

        # Function to compute curvature based on node distance
        def curvature(i,j):
            # small curvature for adjacent nodes, larger for opposite
            angle_diff = abs(i - j)
            return 0.15 * (-1)**(i+j)  # alternate directions to reduce overlap

        # Draw directed edges
        for i,j in directed_edges:
            ax.annotate("",
                        xy=pos[j], xycoords='data',
                        xytext=pos[i], textcoords='data',
                        arrowprops=dict(arrowstyle='-|>',
                                        connectionstyle=f'arc3,rad={curvature(i,j)}',
                                        **arrow_props_base))

        # Draw undirected edges
        for i,j in undirected_edges:
            ax.annotate("",
                        xy=pos[j], xycoords='data',
                        xytext=pos[i], textcoords='data',
                        arrowprops=dict(arrowstyle='-',
                                        connectionstyle=f'arc3,rad={curvature(i,j)}',
                                        **arrow_props_base))

        ax.set_title(title, fontsize=14)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.margins(0.1)  # add some margin so nodes aren't cut off

    plt.tight_layout()
    plt.show()
    
    
    
# -------------------------------------------------------
# Helper: extract a pandas adjacency matrix from causal-learn
# -------------------------------------------------------
def get_adjacency_pc(cg, col_names):
    """
    Extract adjacency matrix from a causal-learn PC CausalGraph.

    causal-learn convention for cg.G.graph[i, j]:
      -1  : i --> j  (arrow tail at j means edge points TO j)
       1  : i <-- j  (arrowhead at i)
      In a CPDAG, an undirected edge i -- j has graph[i,j]=graph[j,i]=-1
      A directed edge i --> j has graph[i,j]=-1 and graph[j,i]=1

    We build adj[src, tgt]=1 to mean src --> tgt.
    Undirected edges are kept as adj[i,j]=adj[j,i]=1.
    """
    g = cg.G.graph          # shape (n, n)
    n = len(col_names)
    adj = pd.DataFrame(0, index=col_names, columns=col_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # directed edge i --> j : g[i,j]==-1 AND g[j,i]==1
            if g[i, j] == -1 and g[j, i] == 1:
                adj.iloc[i, j] = 1   # src=i, tgt=j
            # undirected edge i -- j : g[i,j]==-1 AND g[j,i]==-1
            elif g[i, j] == -1 and g[j, i] == -1:
                adj.iloc[i, j] = 1
    return adj


def get_adjacency_ges(record, col_names):
    """
    Extract adjacency matrix from a causal-learn GES result dict.
    record['G'] is a GeneralGraph; same graph encoding as PC above.
    """
    g = record['G'].graph
    n = len(col_names)
    adj = pd.DataFrame(0, index=col_names, columns=col_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if g[i, j] == -1 and g[j, i] == 1:
                adj.iloc[i, j] = 1
            elif g[i, j] == -1 and g[j, i] == -1:
                adj.iloc[i, j] = 1
    return adj
