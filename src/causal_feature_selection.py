import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from src.ci_test import ci_test

# -------------------------------
# HITON-PC
# -------------------------------
def hiton_pc(data, target, alpha=0.05, ci_method="partial"):
    variables = [v for v in data.columns if v != target]
    assoc = {}
    for v in variables:
        if ci_method == "partial":
            corr, _ = pearsonr(data[v], data[target])
            assoc[v] = abs(corr)
        else:
            corr, _ = pearsonr(data[v], data[target])
            assoc[v] = abs(corr)
    ordered = sorted(variables, key=lambda x: assoc[x], reverse=True)
    PC = []
    for X in ordered:
        independent = False
        for k in range(len(PC) + 1):
            for S in combinations(PC, k):
                p = ci_test(data, X, target, list(S), method=ci_method)
                if p > alpha:
                    independent = True
                    break
            if independent:
                break
        if not independent:
            PC.append(X)
            # backward elimination
            for Y in PC.copy():
                cond_set = [v for v in PC if v != Y]
                for k in range(len(cond_set) + 1):
                    for S in combinations(cond_set, k):
                        p = ci_test(data, Y, target, list(S), method=ci_method)
                        if p > alpha:
                            PC.remove(Y)
                            break
                    else:
                        continue
                    break
    return PC

# -------------------------------
# HITON-MB
# -------------------------------
def hiton_mb(data, target, alpha=0.05, ci_method="partial"):
    PC = hiton_pc(data, target, alpha, ci_method)
    MB = set(PC)
    variables = [v for v in data.columns if v != target]
    for X in PC:
        PC_X = hiton_pc(data, X, alpha, ci_method)
        for Y in PC_X:
            if Y == target or Y in MB:
                continue
            S = set(PC) - {X}
            p = ci_test(data, target, Y, list(S | {X}), method=ci_method)
            if p <= alpha:
                MB.add(Y)
    return list(MB)




# -------------------------------
# IAMB (Incremental Association Markov Blanket)
# -------------------------------
def iamb(data, target, alpha=0.05, ci_method="partial"):
    variables = [v for v in data.columns if v != target]
    MB = []

    # Forward Phase: Add variables to MB
    added = True
    while added:
        assoc = {}
        for X in variables:
            if X in MB:
                continue
            p = ci_test(data, X, target, MB, method=ci_method)
            assoc[X] = 1 - p  # higher association => smaller p-value
        if not assoc:
            break
        X_max = max(assoc, key=assoc.get)
        if assoc[X_max] > (1 - alpha):
            MB.append(X_max)
        else:
            added = False
            break

    # Backward Phase: Remove false positives
    for X in MB.copy():
        S = [v for v in MB if v != X]
        p = ci_test(data, X, target, S, method=ci_method)
        if p > alpha:
            MB.remove(X)

    return MB


# -------------------------------
# MMPC (Max-Min Parents and Children)
# -------------------------------
def mmpc(data, target, alpha=0.05, ci_method="partial"):
    variables = [v for v in data.columns if v != target]
    PC = []

    # Forward Phase: Max-Min heuristic
    added = True
    while added:
        scores = {}
        for X in variables:
            if X in PC:
                continue
            # compute min p-value over all subsets of PC
            if not PC:
                p = ci_test(data, X, target, [], method=ci_method)
                min_p = p
            else:
                min_p = min(ci_test(data, X, target, list(S), method=ci_method)
                            for k in range(len(PC)+1)
                            for S in combinations(PC, k))
            scores[X] = min_p

        # pick variable with smallest min p-value
        if not scores:
            break
        X_star = min(scores, key=scores.get)
        if scores[X_star] <= alpha:
            PC.append(X_star)
        else:
            added = False
            break

    # Backward Phase: remove false positives
    for X in PC.copy():
        S = [v for v in PC if v != X]
        for k in range(len(S)+1):
            for subset in combinations(S, k):
                p = ci_test(data, X, target, list(subset), method=ci_method)
                if p > alpha:
                    PC.remove(X)
                    break
            else:
                continue
            break

    return PC


# -------------------------------
# MB-GES (Markov Blanket - Greedy Equivalence Search) with custom regressor
# -------------------------------
def mb_ges(data, target, score_method="bic", regressor=None):
    if regressor is None:
        regressor = LinearRegression()

    variables = [v for v in data.columns if v != target]
    MB = []
    
    def score(Y, X):
        """Compute score of regression Y ~ X using chosen regressor"""
        if not X:
            residual = Y - Y.mean()
            rss = np.sum(residual ** 2)
            n = len(Y)
            k = 0
        else:
            model = regressor.__class__(**getattr(regressor, 'get_params', lambda: {})())
            model.fit(data[X], Y)
            pred = model.predict(data[X])
            residual = Y - pred
            rss = np.sum(residual ** 2)
            n = len(Y)
            k = len(X)
        if score_method == "bic":
            return n * np.log(rss / n + 1e-10) + k * np.log(n)
        elif score_method == "aic":
            return n * np.log(rss / n + 1e-10) + 2 * k
        else:
            raise ValueError("Unsupported score_method")
    
    # Forward Phase: Greedy addition
    added = True
    while added:
        best_score = score(data[target], MB)
        best_var = None
        for X in variables:
            if X in MB:
                continue
            s = score(data[target], MB + [X])
            if s < best_score:  # lower score is better
                best_score = s
                best_var = X
        if best_var:
            MB.append(best_var)
        else:
            added = False
    
    # Backward Phase: Prune unnecessary variables
    for X in MB.copy():
        s = score(data[target], [v for v in MB if v != X])
        s_full = score(data[target], MB)
        if s <= s_full:
            MB.remove(X)
    
    return MB


# -------------------------------
# RESIT (ANM-based Markov Blanket Discovery)
# -------------------------------

def resit_mb(data, target, alpha=0.05, ci_method="partial", regressor=None, return_parents=False):
    if regressor is None:
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
    
    variables = [v for v in data.columns if v != target]
    MB = []

    # -------------------------
    # Forward Phase: Grow MB
    # -------------------------
    added = True
    while added:
        added = False
        best_var = None
        best_pval = 0
        
        for X in variables:
            if X in MB:
                continue
            
            # Fit target on current MB + candidate X
            if MB:
                model = regressor.__class__(**getattr(regressor, 'get_params', lambda: {})())
                model.fit(data[MB + [X]], data[target])
                residual = data[target] - model.predict(data[MB + [X]])
            else:
                model = regressor.__class__(**getattr(regressor, 'get_params', lambda: {})())
                model.fit(data[[X]], data[target])
                residual = data[target] - model.predict(data[[X]])
            
            # Test independence of residual with candidate
            pval = ci_test(data.assign(residual=residual), 'residual', X, cond_set=MB, method=ci_method)
            
            if pval > best_pval:
                best_pval = pval
                best_var = X
        
        if best_var and best_pval > alpha:
            MB.append(best_var)
            added = True

    # -------------------------
    # Backward Phase: Shrink MB
    # -------------------------
    for X in MB.copy():
        S = [v for v in MB if v != X]
        if S:
            model = regressor.__class__(**getattr(regressor, 'get_params', lambda: {})())
            model.fit(data[S], data[target])
            residual = data[target] - model.predict(data[S])
        else:
            residual = data[target] - data[target].mean()
        
        pval = ci_test(data.assign(residual=residual), 'residual', X, cond_set=S, method=ci_method)
        if pval <= alpha:
            MB.remove(X)
    
    # -------------------------
    # Optional: Extract parents only
    # -------------------------
    if return_parents:
        parents = []
        for X in MB:
            # Test conditional independence given all other MB members except X
            S = [v for v in MB if v != X]
            pval = ci_test(data, target, X, cond_set=S, method=ci_method)
            if pval <= alpha:
                parents.append(X)
        return MB, parents
    
    return MB
