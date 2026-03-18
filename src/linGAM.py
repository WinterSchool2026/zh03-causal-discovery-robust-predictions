import numpy as np

from sklearn.linear_model import LinearRegression

# Priority: causal-learn  >  lingam  >  from-scratch

def _get_lingam_backend():
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM as _DL
        return 'causal-learn', _DL
    except ImportError:
        pass
    try:
        from lingam import DirectLiNGAM as _DL
        return 'lingam', _DL
    except ImportError:
        pass
    return 'scratch', None

_LINGAM_BACKEND, _DirectLiNGAM_cls = _get_lingam_backend()
print(f'LinGAM backend in use: {_LINGAM_BACKEND}')


def _negentropy(x):
    """Non-Gaussianity score via the logcosh approximation (Hyvärinen 1998).
    Higher value = more non-Gaussian = more likely to be a root / exogenous variable.
    """
    x = x / (x.std() + 1e-10)                               # standardise
    return (np.mean(np.log(np.cosh(x))) - np.log(np.cosh(1.0))) ** 2


def _direct_lingam_scratch(X_arr, threshold=0.10):
    """DirectLiNGAM — numpy/sklearn implementation.
    Returns causal_order (int list) and B matrix (same convention as the libraries).
    B[i, j] != 0  =>  variable j is a direct cause of variable i.
    """
    X, X_res = X_arr.astype(float).copy(), X_arr.astype(float).copy()
    p = X.shape[1]
    remaining, causal_order_idx = list(range(p)), []

    # ── Stage 1: iteratively peel off the most exogenous variable ────────
    for _ in range(p):
        if len(remaining) == 1:
            causal_order_idx.append(remaining[0]); break
        # Score: negentropy of residual after removing linear effect of all others
        scores = {}
        for r in remaining:
            others = [o for o in remaining if o != r]
            xr = X_res[:, r].copy()
            if others:
                lr = LinearRegression(fit_intercept=True).fit(X_res[:, others], xr)
                resid = xr - lr.predict(X_res[:, others])
            else:
                resid = xr - xr.mean()
            scores[r] = _negentropy(resid)
        best = max(scores, key=scores.get)   # most exogenous
        causal_order_idx.append(best)
        remaining.remove(best)
        # Regress `best` out of all remaining variables
        xb = X_res[:, best].reshape(-1, 1)
        for r in remaining:
            lr2 = LinearRegression(fit_intercept=True).fit(xb, X_res[:, r])
            X_res[:, r] -= lr2.predict(xb).ravel()

    # ── Stage 2: OLS pruning on original data ────────────────────────────
    B = np.zeros((p, p))
    for k, i in enumerate(causal_order_idx):
        preds = causal_order_idx[:k]
        if preds:
            lr3 = LinearRegression(fit_intercept=True).fit(X[:, preds], X[:, i])
            for j_pos, j in enumerate(preds):
                B[i, j] = lr3.coef_[j_pos]
    B[np.abs(B) <= threshold] = 0.0
    np.fill_diagonal(B, 0.0)
    return causal_order_idx, B


def direct_lingam(df_sub, target, threshold=0.10):
    """
    Run DirectLiNGAM on df_sub (HITON-PC columns + target).

    Parameters
    ----------
    df_sub    : pd.DataFrame — PC set columns plus the target variable
    target    : str          — name of the target column
    threshold : float        — coefficients below this value are pruned to zero

    Returns
    -------
    causal_order : list[str]    — variable names in inferred causal order
    B_pruned     : ndarray(p,p) — B[i,j] != 0  =>  variable j causes variable i
    parents      : list[str]    — inferred parents of `target`
    """
    cols  = list(df_sub.columns)
    X_arr = df_sub.values.astype(float)

    if _LINGAM_BACKEND in ('causal-learn', 'lingam'):
        model = _DirectLiNGAM_cls()
        model.fit(X_arr)
        causal_order_idx = list(model.causal_order_)
        B_pruned = model.adjacency_matrix_.copy()
        B_pruned[np.abs(B_pruned) <= threshold] = 0.0
        np.fill_diagonal(B_pruned, 0.0)
    else:
        causal_order_idx, B_pruned = _direct_lingam_scratch(X_arr, threshold)

    causal_order = [cols[i] for i in causal_order_idx]
    t_pos        = causal_order.index(target)
    t_var_idx    = causal_order_idx[t_pos]
    parents = [
        cols[causal_order_idx[k]]
        for k in range(t_pos)
        if B_pruned[t_var_idx, causal_order_idx[k]] != 0
    ]
    return causal_order, B_pruned, parents
