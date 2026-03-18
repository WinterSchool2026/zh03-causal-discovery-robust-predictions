import numpy as np
from scipy.linalg import eigh
from scipy.stats import gamma

def rbf_kernel(X, sigma=None):
    """
    Compute the RBF (Gaussian) kernel matrix.
    X: n x d
    sigma: bandwidth (if None, median heuristic)
    """
    sq_dists = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * X.dot(X.T)
    
    if sigma is None:
        # median heuristic
        dist_vals = sq_dists[np.triu_indices(sq_dists.shape[0], k=1)]
        sigma = np.sqrt(0.5 * np.median(dist_vals[dist_vals > 0])) + 1e-12
    
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K


def center_kernel(K):
    """
    Center kernel matrix in feature space:
    H = I - 1/n * 11^T
    K_c = HKH
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H.dot(K).dot(H)


def kci_stat(X, Y, Z=None):
    """
    Compute the KCI test statistic and approximate null distribution parameters.
    Returns (statistic, null_alpha, null_beta).
    """

    n = X.shape[0]

    Kx = center_kernel(rbf_kernel(X))
    Ky = center_kernel(rbf_kernel(Y))

    if Z is not None and Z.size > 0:
        Kz = center_kernel(rbf_kernel(Z))

        # residualize Kx and Ky wrt Z
        # Rz = I - Kz*(Kz+λI)^{-1}
        lam = 1e-6  # small regularization
        eigvals, eigvecs = eigh(Kz + lam * np.eye(n))
        inv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
        
        Rz = np.eye(n) - Kz.dot(inv)
        Kx = Rz.dot(Kx).dot(Rz)
        Ky = Rz.dot(Ky).dot(Rz)

    # test statistic (HSIC-like)
    stat = np.trace(Kx.dot(Ky)) / (n ** 2)

    # approximate null distribution via Gamma
    # estimate moments
    mu = np.trace(Kx) * np.trace(Ky) / (n ** 2 * (n - 1))
    var = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / (n ** 4 * (n - 1) * (n + 1))

    alpha = mu**2 / var
    beta = var / mu

    return stat, alpha, beta


def kci_test(data, x, y, cond_set=None):
    """
    Kernel Conditional Independence test.
    Returns an approximate p-value for X ⟂ Y | Z.
    """

    X_vals = data[[x]].to_numpy()
    Y_vals = data[[y]].to_numpy()

    if cond_set is not None:
        Z_vals = data[cond_set].to_numpy()
    else:
        Z_vals = cond_set

    stat, alpha, beta = kci_stat(X_vals, Y_vals, Z_vals)

    # p-value from approximate Gamma
    pvalue = 1 - gamma.cdf(stat, alpha, scale=beta)

    return pvalue