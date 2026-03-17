"""
FRED data download using direct CSV access (no API key needed).

Uses URL pattern: https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}&cosd={START}&coed={END}
"""
import logging

import pandas as pd

from jappelli_experiments.config import (
    FRED_SERIES, FRED_URL_TEMPLATE, SAMPLE_START, SAMPLE_END,
)
from jappelli_experiments.data.cache import save_cache, load_cache

logger = logging.getLogger(__name__)


def download_fred_series(series_id, start=SAMPLE_START, end=SAMPLE_END):
    """
    Download a single FRED series as a DataFrame.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., 'DGS10', 'VIXCLS').
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).

    Returns
    -------
    DataFrame with columns: date, value
    """
    url = FRED_URL_TEMPLATE.format(series=series_id, start=start, end=end)
    logger.info(f"Downloading FRED series {series_id}")

    df = pd.read_csv(url, parse_dates=["DATE"])
    df.columns = ["date", series_id.lower()]

    # FRED uses '.' for missing values
    df[series_id.lower()] = pd.to_numeric(df[series_id.lower()], errors="coerce")

    logger.info(f"  {series_id}: {len(df)} observations, {df['date'].min()} to {df['date'].max()}")
    return df


def download_all_fred(start=SAMPLE_START, end=SAMPLE_END, use_cache=True):
    """
    Download all configured FRED series and merge into a single panel.

    Parameters
    ----------
    start : str
        Start date.
    end : str
        End date.
    use_cache : bool
        If True, try loading from cache first.

    Returns
    -------
    DataFrame with date index and one column per series.
    """
    if use_cache:
        cached = load_cache("fred_all")
        if cached is not None:
            return cached

    merged = None

    for series_id in FRED_SERIES:
        try:
            df = download_fred_series(series_id, start, end)
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="date", how="outer")
        except Exception as e:
            logger.error(f"Failed to download {series_id}: {e}")
            continue

    if merged is not None:
        merged = merged.sort_values("date").reset_index(drop=True)
        save_cache(merged, "fred_all")

    return merged


def get_fred_monthly(start=SAMPLE_START, end=SAMPLE_END):
    """
    Get FRED data resampled to month-end frequency.

    Returns
    -------
    DataFrame with month-end date index.
    """
    df = download_all_fred(start, end)
    if df is None:
        return None

    df = df.set_index("date")
    # Resample to month-end, forward-fill for daily series
    monthly = df.resample("ME").last()
    return monthly


def get_risk_free_rate(freq="monthly"):
    """
    Get risk-free rate series.

    Uses 3-month T-bill rate (TB3MS) converted to the requested frequency.

    Returns
    -------
    Series with date index.
    """
    df = download_all_fred()
    if df is None or "tb3ms" not in df.columns:
        raise ValueError("Could not load TB3MS from FRED")

    rf = df.set_index("date")["tb3ms"] / 100  # Convert from percent

    if freq == "monthly":
        rf = rf.resample("ME").last() / 12  # Annualized -> monthly
    elif freq == "quarterly":
        rf = rf.resample("QE").last() / 4

    return rf


def get_recession_dates():
    """
    Get NBER recession dates from USREC series.

    Returns
    -------
    DataFrame with 'start' and 'end' columns for each recession.
    """
    df = download_all_fred()
    if df is None or "usrec" not in df.columns:
        return pd.DataFrame(columns=["start", "end"])

    rec = df[["date", "usrec"]].dropna()
    rec["usrec"] = rec["usrec"].astype(int)

    # Find transitions
    rec["change"] = rec["usrec"].diff()
    starts = rec[rec["change"] == 1]["date"].tolist()
    ends = rec[rec["change"] == -1]["date"].tolist()

    # Handle edge cases
    if rec["usrec"].iloc[0] == 1:
        starts.insert(0, rec["date"].iloc[0])
    if rec["usrec"].iloc[-1] == 1:
        ends.append(rec["date"].iloc[-1])

    n = min(len(starts), len(ends))
    return pd.DataFrame({"start": starts[:n], "end": ends[:n]})
