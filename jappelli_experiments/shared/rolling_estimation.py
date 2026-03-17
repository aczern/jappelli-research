"""
Rolling-window and expanding-window estimation utilities.

Used for time-varying parameter models (C1: time-varying beta_t).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from jappelli_experiments.config import ROLLING_WINDOW_MONTHS


def rolling_ols(data, y_col, x_cols, window=ROLLING_WINDOW_MONTHS,
                add_constant=True, min_periods=None):
    """
    Rolling-window OLS regression.

    Parameters
    ----------
    data : DataFrame
        Time-sorted data.
    y_col : str
        Dependent variable.
    x_cols : list of str
        Independent variables.
    window : int
        Rolling window size (number of observations).
    add_constant : bool
        Add intercept.
    min_periods : int or None
        Minimum observations required. Defaults to window.

    Returns
    -------
    DataFrame with time-varying coefficients and R-squared.
    """
    if min_periods is None:
        min_periods = window

    df = data[[y_col] + x_cols].dropna()
    n = len(df)

    coef_names = (["const"] + x_cols) if add_constant else x_cols
    results = {name: [] for name in coef_names}
    results["r2"] = []
    results["n_obs"] = []
    results["index"] = []

    for i in range(n):
        start = max(0, i - window + 1)
        window_df = df.iloc[start:i + 1]

        if len(window_df) < min_periods:
            for name in coef_names:
                results[name].append(np.nan)
            results["r2"].append(np.nan)
            results["n_obs"].append(len(window_df))
            results["index"].append(df.index[i])
            continue

        y = window_df[y_col].values
        X = window_df[x_cols].values
        if add_constant:
            X = sm.add_constant(X)

        try:
            res = sm.OLS(y, X).fit()
            for j, name in enumerate(coef_names):
                results[name].append(res.params[j])
            results["r2"].append(res.rsquared)
        except Exception:
            for name in coef_names:
                results[name].append(np.nan)
            results["r2"].append(np.nan)

        results["n_obs"].append(len(window_df))
        results["index"].append(df.index[i])

    out = pd.DataFrame(results)
    out.index = results["index"]
    out = out.drop(columns=["index"])
    return out


def expanding_ols(data, y_col, x_cols, min_periods=36, add_constant=True):
    """
    Expanding-window OLS regression (growing window from start).

    Returns
    -------
    DataFrame with time-varying coefficients.
    """
    return rolling_ols(data, y_col, x_cols, window=len(data),
                       add_constant=add_constant, min_periods=min_periods)


def rolling_correlation(x, y, window=ROLLING_WINDOW_MONTHS):
    """
    Compute rolling Pearson correlation between two series.

    Returns
    -------
    Series of rolling correlations.
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    return df["x"].rolling(window).corr(df["y"])


def rolling_sharpe(returns, rf_rate, window=12):
    """
    Compute rolling realized Sharpe ratio.

    Parameters
    ----------
    returns : Series
        Portfolio/market returns.
    rf_rate : Series
        Risk-free rate (same frequency).
    window : int
        Rolling window (number of periods).

    Returns
    -------
    Series of rolling Sharpe ratios.
    """
    excess = returns - rf_rate
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std()
    return mean / std
