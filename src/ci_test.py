import numpy as np
from scipy.linalg import eigh
from scipy.stats import gamma, pearsonr, t as t_dist
from sklearn.linear_model import LinearRegression
from scipy.stats import norm as scipy_norm
import itertools



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
    mu   = np.trace(Kx) * np.trace(Ky) / (n ** 2 * (n - 1))
    var  = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / (n ** 4 * (n - 1) * (n + 1))

    alpha = mu**2 / var
    beta  = var / mu
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
    n = len(data)

    if len(cond_set) == 0:
        r, _ = pearsonr(data[x], data[y])
    else:
        reg_x = LinearRegression().fit(data[cond_set], data[x])
        rx = data[x] - reg_x.predict(data[cond_set])

        reg_y = LinearRegression().fit(data[cond_set], data[y])
        ry = data[y] - reg_y.predict(data[cond_set])

        r, _ = pearsonr(rx, ry)

    # Correct degrees of freedom: n - |cond_set| - 2
    # pearsonr uses df=n-2 internally — we must recompute the p-value ourselves.
    df = n - len(cond_set) - 2
    if df <= 0:
        return 1.0

    # Clamp r to avoid numerical issues in sqrt
    r = np.clip(r, -1 + 1e-10, 1 - 1e-10)
    t_stat = r * np.sqrt(df) / np.sqrt(1 - r**2)
    p = 2 * t_dist.sf(abs(t_stat), df)
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
    
    
def fisher_z_test(df, x, y, cond=None, alpha=0.05):
    """
    Partial-correlation Fisher z-test for X ⊥ Y | cond.
    Returns (r_partial, p_value, is_independent).
    """
    data_np = df.values.astype(float)
    cols = list(df.columns)
    ix, iy = cols.index(x), cols.index(y)

    if not cond:
        # plain Pearson
        r = np.corrcoef(data_np[:, ix], data_np[:, iy])[0, 1]
    else:
        # partial correlation via matrix inversion
        iz = [cols.index(c) for c in cond]
        idx = [ix, iy] + iz
        sub = np.corrcoef(data_np[:, idx].T)
        try:
            inv = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            return (0.0, 1.0, True)
        r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])

    r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
    n = len(df)
    k = len(cond) if cond else 0
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - k - 3)
    p = 2 * (1 - scipy_norm.cdf(abs(z) / se))
    return (r, p, p > alpha)



def ci_table(df, label):
    vars_ = list(df.columns)
    rows = []

    # Generate all unique pairs of variables
    pairs = list(itertools.combinations(vars_, 2))

    for x, y in pairs:
        # Conditioning set: all other variables except x and y
        cond_vars = [v for v in vars_ if v not in (x, y)]

        # Marginal test (no conditioning)
        r_m, p_m, ind_m = fisher_z_test(df, x, y)
        # Conditional test (condition on all other variables)
        r_c, p_c, ind_c = fisher_z_test(df, x, y, cond=cond_vars if cond_vars else None)

        rows.append({
            'pair': f'{x} ⊥ {y}',
            'r_marginal': f'{r_m:+.3f}',
            'p_marginal': f'{p_m:.3f}',
            'indep_marginal': '✓ indep' if ind_m else '✗ dep',
            'cond_on': ', '.join(cond_vars) if cond_vars else '∅',
            'r_partial': f'{r_c:+.3f}',
            'p_partial': f'{p_c:.3f}',
            'indep_partial': '✓ indep' if ind_c else '✗ dep',
        })

    # Print results
    print(f'\n── {label} ──')
    header = f'{"Pair":<12}  {"Marginal r":>10}  {"p":>6}  {"":<8}  {"Cond on":>10}  {"Partial r":>10}  {"p":>6}  {"":<8}'
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["pair"]:<12}  {r["r_marginal"]:>10}  {r["p_marginal"]:>6}  {r["indep_marginal"]:<8}  '
              f'{r["cond_on"]:>10}  {r["r_partial"]:>10}  {r["p_partial"]:>6}  {r["indep_partial"]:<8}')