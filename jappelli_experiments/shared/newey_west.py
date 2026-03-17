"""
Newey-West (1987) HAC standard errors.

Provides both standalone functions and OLS wrapper with NW correction.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_se(x, max_lag):
    """
    Compute Newey-West HAC standard error for the mean of a time series.

    Parameters
    ----------
    x : array-like
        Time series of estimates (e.g., Fama-MacBeth coefficient series).
    max_lag : int
        Maximum lag for the Bartlett kernel.

    Returns
    -------
    float
        HAC standard error of the mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan

    x_demeaned = x - x.mean()

    # Gamma_0
    gamma_0 = np.dot(x_demeaned, x_demeaned) / n

    # Weighted autocovariances
    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        if j >= n:
            break
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.dot(x_demeaned[j:], x_demeaned[:-j]) / n
        nw_var += 2 * weight * gamma_j

    return np.sqrt(nw_var / n)


def ols_nw(y, X, max_lag, add_constant=True):
    """
    OLS regression with Newey-West HAC standard errors.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X : array-like or DataFrame
        Independent variables.
    max_lag : int
        Maximum lag for NW correction.
    add_constant : bool
        Whether to add an intercept.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    if isinstance(X, pd.DataFrame):
        col_names = list(X.columns)
        X = X.values
    else:
        col_names = None

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if add_constant:
        X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop")
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lag})
    return results


def nw_t_test(x, max_lag):
    """
    Test whether the mean of x is significantly different from zero using NW SE.

    Returns
    -------
    dict with keys: mean, se, t_stat, p_value, n
    """
    from scipy import stats as sp_stats

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mean = x.mean()
    se = newey_west_se(x, max_lag)
    t_stat = mean / se if se > 0 else np.nan
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

    return {"mean": mean, "se": se, "t_stat": t_stat, "p_value": p_value, "n": n}
