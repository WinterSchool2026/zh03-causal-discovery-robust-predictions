import numpy as np
from scipy.linalg import eigh
from scipy.stats import gamma
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


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
