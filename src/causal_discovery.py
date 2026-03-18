from itertools import combinations, permutations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.ci_test import ci_test
from sklearn.decomposition import FastICA



# -------------------------------
# GES (Greedy Equivalence Search) for causal discovery
# -------------------------------
# def ges(data, score_method="bic", regressor=None):
#     if regressor is None:
#         regressor = LinearRegression()
        
#     nodes = list(data.columns)
#     G = {node: set() for node in nodes}  # adjacency sets (parents)
    
#     def score(Y, X):
#         """Compute score of regression Y ~ X using chosen regressor"""
#         if not X:
#             residual = Y - Y.mean()
#             rss = np.sum(residual ** 2)
#             n = len(Y)
#             k = 0
#         else:
#             model = regressor.__class__(**getattr(regressor, 'get_params', lambda: {})())
#             model.fit(data[X], Y)
#             pred = model.predict(data[X])
#             residual = Y - pred
#             rss = np.sum(residual ** 2)
#             n = len(Y)
#             k = len(X)
#         if score_method == "bic":
#             return n * np.log(rss / n + 1e-10) + k * np.log(n)
#         elif score_method == "aic":
#             return n * np.log(rss / n + 1e-10) + 2 * k
#         else:
#             raise ValueError("Unsupported score_method")
    
#     # Forward Phase: Add edges greedily if score improves
#     added = True
#     while added:
#         added = False
#         best_improvement = 0
#         best_edge = None
        
#         for X, Y in combinations(nodes, 2):
#             for direction in [(X, Y), (Y, X)]:
#                 src, tgt = direction
#                 if src in G[tgt]:
#                     continue  # skip existing edge
#                 current_score = score(data[tgt], list(G[tgt]))
#                 new_score = score(data[tgt], list(G[tgt]) + [src])
#                 improvement = current_score - new_score
#                 if improvement > best_improvement:
#                     best_improvement = improvement
#                     best_edge = (src, tgt)
        
#         if best_edge:
#             src, tgt = best_edge
#             G[tgt].add(src)
#             added = True
    
#     # Backward Phase: Remove edges if score improves
#     for tgt in nodes:
#         for src in list(G[tgt]):
#             current_score = score(data[tgt], list(G[tgt]))
#             new_score = score(data[tgt], [v for v in G[tgt] if v != src])
#             if new_score <= current_score:
#                 G[tgt].remove(src)
    
#     return G

def ges(data, score_method="bic", regressor=None):
    if regressor is None:
        regressor = LinearRegression()

    nodes = list(data.columns)

    def score(Y, X):
        """Compute BIC/AIC score of regression Y ~ X."""
        if not X:
            residual = Y - Y.mean()
            rss = np.sum(residual ** 2)
            n, k = len(Y), 0
        else:
            model = regressor.__class__(**getattr(regressor, 'get_params', lambda: {})())
            model.fit(data[X], Y)
            residual = Y - model.predict(data[X])
            rss = np.sum(residual ** 2)
            n, k = len(Y), len(X)
        if score_method == "bic":
            return n * np.log(rss / n + 1e-10) + k * np.log(n)
        elif score_method == "aic":
            return n * np.log(rss / n + 1e-10) + 2 * k
        else:
            raise ValueError("Unsupported score_method")

    # -------------------------------------------------------
    # Phase 1 (Forward): greedily add edges that improve score
    # G[tgt] stores the *parent set* of tgt (directed: src → tgt)
    # -------------------------------------------------------
    G = {node: set() for node in nodes}

    added = True
    while added:
        added = False
        best_improvement = 0
        best_edge = None

        for X, Y in combinations(nodes, 2):
            for src, tgt in [(X, Y), (Y, X)]:
                if src in G[tgt]:
                    continue                        # edge already exists
                current_score = score(data[tgt], list(G[tgt]))
                new_score    = score(data[tgt], list(G[tgt]) + [src])
                improvement  = current_score - new_score
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_edge = (src, tgt)

        if best_edge:
            src, tgt = best_edge
            G[tgt].add(src)
            added = True

    # -------------------------------------------------------
    # Phase 2 (Backward): remove edges that do not hurt score
    # -------------------------------------------------------
    for tgt in nodes:
        for src in list(G[tgt]):
            current_score = score(data[tgt], list(G[tgt]))
            new_score     = score(data[tgt], [v for v in G[tgt] if v != src])
            if new_score <= current_score:
                G[tgt].remove(src)

    # -------------------------------------------------------
    # Build an *undirected* skeleton from the parent sets.
    # At this point G encodes a DAG found by greedy search;
    # we now re-orient it properly via V-structures + Meek.
    # -------------------------------------------------------
    skeleton = {node: set() for node in nodes}
    for tgt in nodes:
        for src in G[tgt]:
            skeleton[src].add(tgt)
            skeleton[tgt].add(src)

    # sep_set: for GES we infer separation sets from the skeleton —
    # two non-adjacent nodes are separated by their common neighbours.
    sep_set = {}
    for X in nodes:
        for Y in nodes:
            if X == Y or Y in skeleton[X]:
                continue
            sep_set[(X, Y)] = skeleton[X] & skeleton[Y]

    # -------------------------------------------------------
    # Phase 3: V-structure (collider) orientation
    # Unshielded triple X — Y — Z, Y ∉ sep_set(X, Z)  ⟹  X → Y ← Z
    # -------------------------------------------------------
    directed  = {node: set() for node in nodes}   # directed[u] = {v : u → v}
    undirected = {node: set(skeleton[node]) for node in nodes}

    for Y in nodes:
        adj_Y = list(undirected[Y])
        for X, Z in combinations(adj_Y, 2):
            if Z in undirected[X]:                 # shielded triple — skip
                continue
            if Y not in sep_set.get((X, Z), set()):
                # Orient X → Y and Z → Y
                directed[X].add(Y);  directed[Z].add(Y)
                undirected[X].discard(Y);  undirected[Y].discard(X)
                undirected[Z].discard(Y);  undirected[Y].discard(Z)

    # -------------------------------------------------------
    # Phase 4: Meek rules — propagate orientations for consistency
    # -------------------------------------------------------
    def is_adjacent(u, v):
        return v in undirected[u] or v in directed[u] or u in directed[v]

    def orient(u, v):
        """Orient undirected edge u — v  as  u → v."""
        undirected[u].discard(v)
        undirected[v].discard(u)
        directed[u].add(v)

    changed = True
    while changed:
        changed = False
        for B in nodes:

            # R1: A → B — C,  A not adj C   ⟹   B → C
            for A in nodes:
                if B not in directed[A]:
                    continue
                for C in list(undirected[B]):
                    if not is_adjacent(A, C):
                        orient(B, C)
                        changed = True

            # R2: A — B,  A → C → B   ⟹   A → B  (no directed cycle)
            for C in nodes:
                if B not in directed[C]:
                    continue
                for A in list(undirected[B]):
                    if C in directed[A]:
                        orient(A, B)
                        changed = True

            # R3: A → Mid ← C,  A — B,  C — B,  Mid — B,  A not adj C   ⟹   Mid → B
            for Mid in list(undirected[B]):
                parents_Mid = [u for u in nodes if Mid in directed[u]]
                for A, C in combinations(parents_Mid, 2):
                    if A not in undirected[B] or C not in undirected[B]:
                        continue
                    if is_adjacent(A, C):
                        continue
                    orient(Mid, B)
                    changed = True
                    break

            # R4: A — B,  A — C → Mid → B,  A not adj Mid   ⟹   A → B
            for A in list(undirected[B]):
                for C in list(undirected[A]):
                    if C == B:
                        continue
                    for Mid in directed[C]:
                        if B not in directed[Mid]:
                            continue
                        if is_adjacent(A, Mid):
                            continue
                        orient(A, B)
                        changed = True

    # -------------------------------------------------------
    # Return oriented adjacency dict  result[node] = set of parents
    # Directed  A → B  →  result[B] contains A  (but not vice-versa)
    # Undirected A — B  →  result[A] contains B  AND  result[B] contains A
    # -------------------------------------------------------
    result = {node: set() for node in nodes}
    for src in nodes:
        for tgt in directed[src]:
            result[tgt].add(src)
    for node in nodes:
        for nb in undirected[node]:
            result[node].add(nb)

    return result

# -------------------------------
# PC Algorithm for causal discovery
# -------------------------------
# def pc_alg(data, alpha=0.05, ci_method="partial"):
#     nodes = list(data.columns)
#     # Start with a complete undirected graph
#     G = {node: set(nodes) - {node} for node in nodes}
    
#     l = 0  # size of conditioning set
#     cont = True
#     while cont:
#         cont = False
#         for X in nodes:
#             neighbors = list(G[X])
#             if len(neighbors) < l:
#                 continue
#             for Y in neighbors:
#                 adj_X = list(G[X] - {Y})
#                 if len(adj_X) < l:
#                     continue
#                 for S in combinations(adj_X, l):
#                     p = ci_test(data, X, Y, list(S), method=ci_method)
#                     if p > alpha:  # X ⊥ Y | S → remove edge
#                         G[X].remove(Y)
#                         G[Y].remove(X)
#                         cont = True
#                         break  # no need to test other S
#         l += 1
    
#     return G


def pc_alg(data, alpha=0.05, ci_method="partial", return_ci=False):
    nodes = list(data.columns)

    # Start with a complete undirected graph
    G = {node: set(nodes) - {node} for node in nodes}

    # -------------------------------------------------------
    # Phase 1: Skeleton discovery
    # -------------------------------------------------------
    sep_set   = {(X, Y): set() for X in nodes for Y in nodes if X != Y}
    ci_strings = set()   # "X _||_ Y | {Z, ...}" for every accepted independence

    def _ci_str(X, Y, S):
        cond = "{ " + ", ".join(sorted(str(s) for s in S)) + " }" if S else "\u2205"
        return f"{X} _||_ {Y} | {cond}"

    l = 0
    cont = True
    while cont:
        cont = False
        for X in nodes:
            for Y in list(G[X]):          # always fresh
                if Y not in G[X]:         # may have been removed this pass
                    continue
                adj_X = list(G[X] - {Y}) # current neighbours of X, excluding Y
                if len(adj_X) < l:
                    continue
                for S in combinations(adj_X, l):
                    p = ci_test(data, X, Y, list(S), method=ci_method)
                    if p > alpha:         # X ⊥ Y | S  →  remove edge
                        G[X].discard(Y)
                        G[Y].discard(X)
                        sep_set[(X, Y)] = set(S)
                        sep_set[(Y, X)] = set(S)
                        ci_strings.add(_ci_str(X, Y, S))
                        cont = True
                        break             # no need to test other S for this pair
        l += 1

    # -------------------------------------------------------
    # Phase 2: V-structure (collider) orientation
    # Unshielded triple X — Y — Z, Y not in sep_set(X,Z)  =>  X -> Y <- Z
    # -------------------------------------------------------
    directed   = {node: set() for node in nodes}  # directed[u] = {v : u -> v}

    for Y in nodes:
        adj_Y = list(G[Y])
        for X, Z in combinations(adj_Y, 2):
            if Z in G[X]:                          # shielded triple — skip
                continue
            if Y not in sep_set.get((X, Z), set()):
                directed[X].add(Y);  directed[Z].add(Y)
                G[X].discard(Y);     G[Y].discard(X)
                G[Z].discard(Y);     G[Y].discard(Z)

    # -------------------------------------------------------
    # Phase 3: Meek rules — propagate orientations
    # G[u]        = undirected neighbours of u
    # directed[u] = {v : u -> v}
    # -------------------------------------------------------
    def is_adjacent(u, v):
        return v in G[u] or v in directed[u] or u in directed[v]

    def orient(u, v):
        """Orient undirected edge u — v  as  u -> v."""
        G[u].discard(v);  G[v].discard(u)
        directed[u].add(v)

    changed = True
    while changed:
        changed = False
        for B in nodes:

            # R1: A -> B — C,  A not adj C   =>  B -> C
            for A in list(directed):
                if B not in directed[A]:
                    continue
                for C in list(G[B]):
                    if not is_adjacent(A, C):
                        orient(B, C);  changed = True

            # R2: A — B,  A -> C -> B   =>  A -> B  (no directed cycle)
            for C in list(directed):
                if B not in directed[C]:
                    continue
                for A in list(G[B]):
                    if B in directed[A]:
                        continue
                    if C in directed[A]:
                        orient(A, B);  changed = True

            # R3: A -> D <- C,  A — B,  C — B,  D — B,  A not adj C   =>  D -> B
            for D in list(G[B]):
                parents_of_D = [u for u in directed if D in directed[u]]
                for A, C in combinations(parents_of_D, 2):
                    if A not in G[B] or C not in G[B]:
                        continue
                    if is_adjacent(A, C):
                        continue
                    orient(D, B);  changed = True;  break

            # R4: A — B,  A — C -> Mid -> B,  A not adj Mid   =>  A -> B
            for A in list(G[B]):
                for C in list(G[A]):
                    if C == B:
                        continue
                    for Mid in directed[C]:
                        if B not in directed[Mid]:
                            continue
                        if is_adjacent(A, Mid):
                            continue
                        orient(A, B);  changed = True

    # -------------------------------------------------------
    # Build return value: result[node] = set of parents
    # Directed  A -> B  =>  result[B] contains A (not vice-versa)
    # Undirected A — B  =>  result[A] contains B AND result[B] contains A
    #
    # Safety: strip from G anything already in directed to avoid
    # double-counting edges that were oriented during Meek.
    # -------------------------------------------------------
    result = {node: set() for node in nodes}

    for src in nodes:
        for tgt in directed[src]:
            result[tgt].add(src)      # src is a parent of tgt
            G[src].discard(tgt)       # remove from undirected if still there
            G[tgt].discard(src)

    for node in nodes:
        for nb in G[node]:            # remaining undirected edges
            result[node].add(nb)

    if return_ci:
        return result, ci_strings
    return result


def fci_alg(data, alpha=0.05, ci_method="partial"):
    nodes = list(data.columns)
    
    # Step 1: Skeleton discovery (like PC)
    # Start with a complete undirected graph with circles at endpoints
    G = {node: {n: "o-o" for n in nodes if n != node} for node in nodes}
    
    l = 0  # size of conditioning set
    cont = True
    sep_set = { (X,Y): set() for X in nodes for Y in nodes if X != Y }  # separation sets
    
    while cont:
        cont = False
        for X in nodes:
            neighbors = [n for n in G[X] if G[X][n] != None]
            if len(neighbors) < l:
                continue
            for Y in neighbors:
                adj_X = [n for n in neighbors if n != Y]
                if len(adj_X) < l:
                    continue
                for S in combinations(adj_X, l):
                    p = ci_test(data, X, Y, list(S), method=ci_method)
                    if p > alpha:  # independent → remove edge
                        G[X][Y] = None
                        G[Y][X] = None
                        sep_set[(X,Y)] = set(S)
                        sep_set[(Y,X)] = set(S)
                        cont = True
                        break
        l += 1
    
    # Step 2: Collider orientation
    for X, Y, Z in permutations(nodes, 3):
        if G[X].get(Y) is None or G[Z].get(Y) is None:
            continue
        if G[X][Y] == "o-o" and G[Z][Y] == "o-o":
            # Check if Y not in sep_set(X,Z)
            if Y not in sep_set.get((X,Z), set()):
                G[X][Y] = "->"
                G[Z][Y] = "->"
    
    # Step 3: Propagate orientations (simplified FCI rules)
    changed = True
    while changed:
        changed = False
        for X in nodes:
            for Y in list(G[X].keys()):
                if G[X][Y] is None:
                    continue
                for Z in list(G[Y].keys()):
                    if Z == X or G[Y][Z] is None:
                        continue
                    # Rule R1: X->Y o-o Z → orient Y-Z as Y->Z
                    if G[X][Y] == "->" and G[Y][Z] == "o-o":
                        G[Y][Z] = "->"
                        G[Z][Y] = "<-"  # keep consistent
                        changed = True
    # Remove None entries (edges removed)
    for X in nodes:
        G[X] = {Y: mark for Y, mark in G[X].items() if mark is not None}
    
    return G


def lingam(data):
    nodes = list(data.columns)
    X = data.values
    X = X - X.mean(axis=0)  # center data
    
    # Step 1: ICA to estimate mixing matrix
    ica = FastICA(n_components=X.shape[1], random_state=0)
    S = ica.fit_transform(X)        # estimated sources (independent components)
    A = ica.mixing_                 # mixing matrix
    B_est = np.linalg.inv(A) - np.eye(A.shape[0])  # B = (W - I)
    
    # Step 2: Build DAG adjacency from B_est
    # Keep only positive absolute entries as edges
    G = {node: set() for node in nodes}
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j and abs(B_est[i, j]) > 1e-6:
                G[node_j].add(node_i)  # edge from node_i -> node_j
    return G



def adjacency(G):
    nodes = list(G.keys())
    adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    
    for tgt, parents in G.items():
        for src in parents:
            adj_matrix.loc[src, tgt] = 1  # edge from src -> tgt
    
    return adj_matrix
