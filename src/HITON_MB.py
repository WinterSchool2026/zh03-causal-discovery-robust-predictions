import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.linalg import eigh
from scipy.stats import gamma

# -------------------------------
# KCI Implementation
# -------------------------------
def rbf_kernel(X, sigma=None):
    sq_dists = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * X.dot(X.T)
    if sigma is None:
        dist_vals = sq_dists[np.triu_indices(sq_dists.shape[0], k=1)]
        sigma = np.sqrt(0.5 * np.median(dist_vals[dist_vals > 0])) + 1e-12
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

def center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H.dot(K).dot(H)

def kci_stat(X, Y, Z=None):
    n = X.shape[0]
    Kx = center_kernel(rbf_kernel(X))
    Ky = center_kernel(rbf_kernel(Y))

    if Z is not None and Z.size > 0:
        Kz = center_kernel(rbf_kernel(Z))
        lam = 1e-6
        eigvals, eigvecs = eigh(Kz + lam * np.eye(n))
        inv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
        Rz = np.eye(n) - Kz.dot(inv)
        Kx = Rz.dot(Kx).dot(Rz)
        Ky = Rz.dot(Ky).dot(Rz)

    stat = np.trace(Kx.dot(Ky)) / (n ** 2)

    mu = np.trace(Kx) * np.trace(Ky) / (n ** 2 * (n - 1))
    var = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / (n ** 4 * (n - 1) * (n + 1))

    alpha = mu**2 / var
    beta = var / mu
    return stat, alpha, beta

def kci_test(data, x, y, cond_set):
    X_vals = data[[x]].to_numpy()
    Y_vals = data[[y]].to_numpy()
    Z_vals = None if len(cond_set) == 0 else data[cond_set].to_numpy()
    stat, alpha, beta = kci_stat(X_vals, Y_vals, Z_vals)
    pvalue = 1 - gamma.cdf(stat, alpha, scale=beta)
    return pvalue

# -------------------------------
# Partial Correlation Test
# -------------------------------
def partial_corr_test(data, x, y, cond_set):
    if len(cond_set) == 0:
        _, p = pearsonr(data[x], data[y])
        return p
    reg_x = LinearRegression().fit(data[cond_set], data[x])
    rx = data[x] - reg_x.predict(data[cond_set])
    reg_y = LinearRegression().fit(data[cond_set], data[y])
    ry = data[y] - reg_y.predict(data[cond_set])
    _, p = pearsonr(rx, ry)
    return p

# -------------------------------
# Wrapper to select CI test
# -------------------------------
def ci_test(data, x, y, cond_set, method="partial"):
    if method == "partial":
        return partial_corr_test(data, x, y, cond_set)
    elif method == "kci":
        return kci_test(data, x, y, cond_set)
    else:
        raise ValueError("ci_test method must be 'partial' or 'kci'")

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