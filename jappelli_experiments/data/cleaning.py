"""
Data cleaning utilities: winsorization, outlier handling, merge logic.
"""
import numpy as np
import pandas as pd
from jappelli_experiments.config import WINSORIZE_PCTILE


def winsorize(series, limits=WINSORIZE_PCTILE):
    """
    Winsorize a series at the specified percentiles.

    Parameters
    ----------
    series : Series or array-like
        Data to winsorize.
    limits : tuple of float
        (lower_quantile, upper_quantile), e.g., (0.01, 0.99).

    Returns
    -------
    Series with extreme values clipped.
    """
    s = pd.Series(series)
    lower = s.quantile(limits[0])
    upper = s.quantile(limits[1])
    return s.clip(lower=lower, upper=upper)


def winsorize_panel(panel, cols, limits=WINSORIZE_PCTILE, by_period=None):
    """
    Winsorize multiple columns, optionally within each time period.

    Parameters
    ----------
    panel : DataFrame
    cols : list of str
        Columns to winsorize.
    limits : tuple
        Quantile limits.
    by_period : str or None
        If provided, winsorize within each period (cross-section).

    Returns
    -------
    DataFrame with winsorized columns.
    """
    df = panel.copy()
    for col in cols:
        if by_period:
            df[col] = df.groupby(by_period)[col].transform(
                lambda x: winsorize(x, limits)
            )
        else:
            df[col] = winsorize(df[col], limits)
    return df


def standardize(series):
    """Standardize a series to mean 0, std 1."""
    s = pd.Series(series)
    return (s - s.mean()) / s.std()


def lag_variable(panel, var_col, entity_col, time_col, n_lags=1):
    """
    Create lagged variable within each entity.

    Parameters
    ----------
    panel : DataFrame
    var_col : str
        Variable to lag.
    entity_col : str
        Entity identifier (e.g., 'permno').
    time_col : str
        Time column.
    n_lags : int
        Number of lags.

    Returns
    -------
    Series of lagged values.
    """
    df = panel.sort_values([entity_col, time_col])
    return df.groupby(entity_col)[var_col].shift(n_lags)


def forward_variable(panel, var_col, entity_col, time_col, n_forward=1):
    """Create forward (lead) variable within each entity."""
    df = panel.sort_values([entity_col, time_col])
    return df.groupby(entity_col)[var_col].shift(-n_forward)


def safe_merge(left, right, on, how="inner", validate=None, indicator=False):
    """
    Merge with logging of merge quality.

    Returns
    -------
    DataFrame, dict with merge diagnostics
    """
    n_left = len(left)
    n_right = len(right)

    merged = pd.merge(left, right, on=on, how=how,
                       validate=validate, indicator=True)

    diag = {
        "n_left": n_left,
        "n_right": n_right,
        "n_merged": len(merged),
        "both": (merged["_merge"] == "both").sum(),
        "left_only": (merged["_merge"] == "left_only").sum(),
        "right_only": (merged["_merge"] == "right_only").sum(),
        "match_rate": (merged["_merge"] == "both").sum() / n_left if n_left > 0 else 0,
    }

    if not indicator:
        merged = merged.drop(columns=["_merge"])

    return merged, diag


def to_monthly(df, date_col):
    """
    Convert a date column to month-end frequency.

    Parameters
    ----------
    df : DataFrame
    date_col : str
        Date column to convert.

    Returns
    -------
    DataFrame with date_col rounded to month-end.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0)
    return df


def to_quarterly(df, date_col):
    """Convert a date column to quarter-end frequency."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]) + pd.offsets.QuarterEnd(0)
    return df


def drop_microcaps(df, me_col, threshold_pctile=5):
    """
    Drop stocks below the given market equity percentile (NYSE breakpoints).

    Parameters
    ----------
    df : DataFrame
    me_col : str
        Market equity column.
    threshold_pctile : float
        Percentile cutoff (e.g., 5 for bottom 5%).

    Returns
    -------
    DataFrame with microcaps removed.
    """
    cutoff = df[me_col].quantile(threshold_pctile / 100)
    return df[df[me_col] >= cutoff].copy()
