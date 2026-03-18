import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy.stats import f

def icp(data_list, target, alpha=0.05, regressor=None):
    """
    Invariant Causal Prediction (ICP) using a list of datasets for different environments,
    with a flexible regression model (default: LinearRegression).

    Args:
        data_list: list of pd.DataFrame, each DataFrame is one environment.
        target: str, name of the target variable.
        alpha: significance threshold for F-test.
        regressor: regression model with fit/predict interface (default: LinearRegression).

    Returns:
        direct_causes: list of variables that are invariant causal predictors.
    """
    if regressor is None:
        regressor = LinearRegression

    variables = [v for v in data_list[0].columns if v != target]
    direct_causes = []

    # ----------------------------
    # Check all candidate subsets
    # ----------------------------
    for k in range(1, len(variables) + 1):
        for subset in combinations(variables, k):
            subset = list(subset)
            invariant = True

            # Fit model in each environment
            residuals_all = []
            env_labels = []
            for i, env_data in enumerate(data_list):
                X_env = env_data[subset].values
                y_env = env_data[target].values
                model = regressor()
                model.fit(X_env, y_env)
                residuals = y_env - model.predict(X_env)
                residuals_all.append(residuals)
                env_labels.extend([i] * len(residuals))

            # Stack residuals and test dependence on environment
            residuals_all = np.concatenate(residuals_all)
            env_labels = np.array(env_labels).reshape(-1, 1)

            # Regression of residuals on environment dummy
            env_dummies = pd.get_dummies(env_labels.flatten(), drop_first=True).values
            if env_dummies.shape[1] == 0:
                continue  # only one environment
            env_model = LinearRegression()
            env_model.fit(env_dummies, residuals_all)
            rss_full = np.sum((residuals_all - env_model.predict(env_dummies))**2)
            rss_null = np.sum(residuals_all**2)
            df1 = env_dummies.shape[1]
            df2 = len(residuals_all) - df1 - 1
            F_stat = ((rss_null - rss_full)/df1) / (rss_full/df2 + 1e-10)
            p_val = 1 - f.cdf(F_stat, df1, df2)

            if p_val <= alpha:
                invariant = False

            if invariant:
                direct_causes.extend(subset)

    # Remove duplicates
    direct_causes = list(set(direct_causes))
    return direct_causes