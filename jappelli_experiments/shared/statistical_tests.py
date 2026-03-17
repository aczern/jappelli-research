"""
Statistical tests used across experiments.

ADF, Granger causality, Chow breakpoint, Benjamini-Hochberg FDR correction,
White's heteroscedasticity test, Durbin-Watson, VIF.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def adf_test(series, max_lags=None, regression="c"):
    """
    Augmented Dickey-Fuller unit root test.

    Parameters
    ----------
    series : array-like
        Time series to test.
    max_lags : int or None
        Maximum lags (None = automatic via AIC).
    regression : str
        'c' (constant), 'ct' (constant + trend), 'n' (none).

    Returns
    -------
    dict with adf_stat, p_value, lags_used, n_obs, critical_values, is_stationary
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]

    result = adfuller(x, maxlag=max_lags, regression=regression, autolag="AIC")

    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def chow_test(y, X, break_point):
    """
    Chow (1960) structural break test.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X : array-like
        Regressors (include constant if desired).
    break_point : int
        Index at which to split the sample.

    Returns
    -------
    dict with f_stat, p_value, is_break
    """
    y = np.asarray(y)
    X = np.asarray(X)
    n = len(y)
    k = X.shape[1]

    # Full sample
    res_full = sm.OLS(y, X).fit()
    ssr_full = res_full.ssr

    # Sub-samples
    res1 = sm.OLS(y[:break_point], X[:break_point]).fit()
    res2 = sm.OLS(y[break_point:], X[break_point:]).fit()
    ssr_parts = res1.ssr + res2.ssr

    # F-statistic
    f_stat = ((ssr_full - ssr_parts) / k) / (ssr_parts / (n - 2 * k))
    p_value = 1 - sp_stats.f.cdf(f_stat, k, n - 2 * k)

    return {"f_stat": f_stat, "p_value": p_value, "is_break": p_value < 0.05}


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg (1995) FDR correction for multiple testing.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Desired FDR level.

    Returns
    -------
    DataFrame with original p-values, adjusted p-values, and rejection decisions.
    """
    p = np.asarray(p_values)
    n = len(p)
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # BH adjusted p-values
    adjusted = np.zeros(n)
    adjusted[-1] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))

    # Unsort
    result = np.zeros(n)
    result[sorted_idx] = adjusted

    return pd.DataFrame({
        "p_value": p_values,
        "p_adjusted": result,
        "reject": result < alpha,
    })


def whites_test(residuals, X):
    """
    White's (1980) heteroscedasticity test.

    Returns
    -------
    dict with test_stat, p_value, is_heteroscedastic
    """
    stat, p_value, _, _ = het_white(residuals, X)
    return {"test_stat": stat, "p_value": p_value, "is_heteroscedastic": p_value < 0.05}


def compute_vif(X, col_names=None):
    """
    Compute Variance Inflation Factors for regressors.

    Parameters
    ----------
    X : array-like or DataFrame
        Regressor matrix (should include constant if used in regression).
    col_names : list of str or None
        Names for columns.

    Returns
    -------
    DataFrame with variable names and VIF values.
    """
    X = np.asarray(X)
    if col_names is None:
        col_names = [f"X{i}" for i in range(X.shape[1])]

    vifs = []
    for i in range(X.shape[1]):
        vifs.append(variance_inflation_factor(X, i))

    return pd.DataFrame({"variable": col_names, "vif": vifs})


def durbin_watson_test(residuals):
    """Compute Durbin-Watson statistic for autocorrelation."""
    return durbin_watson(residuals)


def regression_diagnostics(results, X):
    """
    Run standard diagnostic battery on an OLS result.

    Parameters
    ----------
    results : statsmodels RegressionResults
    X : array-like
        Regressor matrix used in estimation.

    Returns
    -------
    dict of diagnostic results
    """
    resid = results.resid
    diag = {
        "n_obs": int(results.nobs),
        "r2": results.rsquared,
        "r2_adj": results.rsquared_adj,
        "durbin_watson": durbin_watson_test(resid),
        "white_test": whites_test(resid, X),
    }

    # VIF (skip constant)
    X_arr = np.asarray(X)
    if X_arr.shape[1] > 1:
        diag["vif"] = compute_vif(X_arr)

    return diag
