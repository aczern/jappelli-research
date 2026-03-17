"""
Portfolio sorts: single, double, and conditional.

Computes equal- and value-weighted portfolio returns, long-short spreads,
and risk-adjusted alphas.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from jappelli_experiments.config import NW_LAGS_MONTHLY


def single_sort(panel, time_col, ret_col, sort_col, n_ports=5,
                weight_col=None, lag_sort=True):
    """
    Single-sort portfolio construction.

    Parameters
    ----------
    panel : DataFrame
        Stock-month panel.
    time_col : str
        Time period column.
    ret_col : str
        Return column.
    sort_col : str
        Variable to sort on.
    n_ports : int
        Number of portfolios (quantiles).
    weight_col : str or None
        Column for value-weighting. None = equal-weight.
    lag_sort : bool
        If True, sort on lagged sort_col (avoid look-ahead).

    Returns
    -------
    DataFrame with portfolio returns per period.
    """
    df = panel[[time_col, ret_col, sort_col]].copy()
    if weight_col is not None:
        df["weight"] = panel[weight_col]

    if lag_sort:
        df[sort_col] = df.groupby(level=0)[sort_col].shift(1) if isinstance(df.index, pd.MultiIndex) else df[sort_col]

    df = df.dropna(subset=[sort_col, ret_col])

    # Assign portfolio ranks within each period
    df["port"] = df.groupby(time_col)[sort_col].transform(
        lambda x: pd.qcut(x, n_ports, labels=False, duplicates="drop") + 1
    )

    if weight_col is not None:
        # Value-weighted returns
        def vw_ret(g):
            w = g["weight"] / g["weight"].sum()
            return (g[ret_col] * w).sum()
        port_rets = df.groupby([time_col, "port"]).apply(vw_ret).unstack("port")
    else:
        # Equal-weighted returns
        port_rets = df.groupby([time_col, "port"])[ret_col].mean().unstack("port")

    port_rets.columns = [f"Q{int(c)}" for c in port_rets.columns]

    # Long-short spread (high minus low)
    port_rets["LS"] = port_rets[f"Q{n_ports}"] - port_rets["Q1"]

    return port_rets


def double_sort(panel, time_col, ret_col, sort_col1, sort_col2,
                n1=5, n2=5, weight_col=None, conditional=True):
    """
    Double-sort portfolio construction.

    Parameters
    ----------
    conditional : bool
        If True, sort on sort_col2 within sort_col1 bins (conditional).
        If False, independent double sort.

    Returns
    -------
    DataFrame with portfolio returns indexed by (time, port1, port2).
    """
    df = panel[[time_col, ret_col, sort_col1, sort_col2]].copy()
    if weight_col is not None:
        df["weight"] = panel[weight_col]
    df = df.dropna()

    # First sort
    df["port1"] = df.groupby(time_col)[sort_col1].transform(
        lambda x: pd.qcut(x, n1, labels=False, duplicates="drop") + 1
    )

    if conditional:
        # Second sort within first sort bins
        df["port2"] = df.groupby([time_col, "port1"])[sort_col2].transform(
            lambda x: pd.qcut(x, n2, labels=False, duplicates="drop") + 1
            if len(x) >= n2 else pd.Series(np.nan, index=x.index)
        )
    else:
        # Independent second sort
        df["port2"] = df.groupby(time_col)[sort_col2].transform(
            lambda x: pd.qcut(x, n2, labels=False, duplicates="drop") + 1
        )

    df = df.dropna(subset=["port1", "port2"])

    if weight_col is not None:
        def vw_ret(g):
            w = g["weight"] / g["weight"].sum()
            return (g[ret_col] * w).sum()
        port_rets = df.groupby([time_col, "port1", "port2"]).apply(vw_ret)
    else:
        port_rets = df.groupby([time_col, "port1", "port2"])[ret_col].mean()

    return port_rets.unstack(["port1", "port2"])


def portfolio_alpha(port_returns, factor_returns, factor_cols, nw_lags=NW_LAGS_MONTHLY):
    """
    Compute risk-adjusted alpha for a portfolio return series.

    Parameters
    ----------
    port_returns : Series
        Portfolio excess returns (time-indexed).
    factor_returns : DataFrame
        Factor returns (same time index).
    factor_cols : list of str
        Factor column names (e.g., ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']).
    nw_lags : int
        Newey-West lags.

    Returns
    -------
    dict with alpha, se, t_stat, p_value, r2
    """
    merged = pd.merge(
        port_returns.rename("port_ret"),
        factor_returns[factor_cols],
        left_index=True, right_index=True, how="inner"
    ).dropna()

    X = sm.add_constant(merged[factor_cols].values)
    y = merged["port_ret"].values

    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    return {
        "alpha": res.params[0],
        "se": res.bse[0],
        "t_stat": res.tvalues[0],
        "p_value": res.pvalues[0],
        "r2": res.rsquared,
        "n_obs": int(res.nobs),
    }
