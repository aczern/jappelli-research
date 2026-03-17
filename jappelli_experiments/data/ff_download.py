"""
Download Fama-French factor data from Ken French's Data Library.

Downloads factor returns (FF5 + momentum) at monthly and daily frequency.
"""
import io
import logging
import zipfile

import pandas as pd

from jappelli_experiments.config import FF_URL_BASE, FF_DATASETS
from jappelli_experiments.data.cache import save_cache, load_cache

logger = logging.getLogger(__name__)


def _download_ff_csv(dataset_key):
    """
    Download a Fama-French dataset CSV from the Ken French website.

    Parameters
    ----------
    dataset_key : str
        Key from config.FF_DATASETS.

    Returns
    -------
    str : CSV content
    """
    filename = FF_DATASETS[dataset_key]
    url = FF_URL_BASE + filename
    logger.info(f"Downloading {dataset_key} from {url}")

    import urllib.request
    response = urllib.request.urlopen(url)
    zf = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = zf.namelist()[0]
    return zf.read(csv_name).decode("utf-8")


def _parse_ff5(csv_text, freq="monthly"):
    """Parse FF5 factor CSV into a clean DataFrame."""
    lines = csv_text.strip().split("\n")

    # Find the header row (contains "Mkt-RF" or similar)
    start_idx = 0
    for i, line in enumerate(lines):
        if "Mkt-RF" in line or "Mkt" in line:
            start_idx = i
            break

    # Read from header to first blank line (monthly data section)
    data_lines = [lines[start_idx]]
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or not line[0].isdigit():
            # Check if this is really a data line
            if len(line) < 5:
                break
        data_lines.append(line)

    # Parse
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # First column is date
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date_str"})
    df["date_str"] = df["date_str"].astype(str).str.strip()

    # Parse date
    if freq == "monthly":
        # Format: YYYYMM
        df = df[df["date_str"].str.len() == 6].copy()
        df["date"] = pd.to_datetime(df["date_str"], format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        # Format: YYYYMMDD
        df = df[df["date_str"].str.len() == 8].copy()
        df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%d")

    df = df.drop(columns=["date_str"])

    # Convert to numeric (values are in percent)
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100

    return df.set_index("date").sort_index()


def _parse_momentum(csv_text, freq="monthly"):
    """Parse momentum factor CSV."""
    lines = csv_text.strip().split("\n")

    start_idx = 0
    for i, line in enumerate(lines):
        if "Mom" in line or len(line.strip().split(",")) >= 2:
            # Check if next lines have data
            if i + 1 < len(lines) and lines[i + 1].strip()[0:1].isdigit():
                start_idx = i
                break

    data_lines = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            break
        data_lines.append(line)

    if not data_lines:
        # Fallback: try reading the whole thing
        df = pd.read_csv(io.StringIO(csv_text), skiprows=range(13), skipinitialspace=True)
    else:
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), skipinitialspace=True)

    df.columns = df.columns.str.strip()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date_str"})
    df["date_str"] = df["date_str"].astype(str).str.strip()

    if freq == "monthly":
        df = df[df["date_str"].str.len() == 6].copy()
        df["date"] = pd.to_datetime(df["date_str"], format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df = df[df["date_str"].str.len() == 8].copy()
        df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%d")

    df = df.drop(columns=["date_str"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
            df = df.rename(columns={col: "Mom"})

    return df.set_index("date").sort_index()


def get_ff5_monthly(use_cache=True):
    """
    Get Fama-French 5 factors at monthly frequency.

    Returns
    -------
    DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    """
    if use_cache:
        cached = load_cache("ff5_monthly")
        if cached is not None:
            return cached

    csv_text = _download_ff_csv("ff5_monthly")
    df = _parse_ff5(csv_text, freq="monthly")

    save_cache(df.reset_index(), "ff5_monthly")
    return df


def get_ff5_daily(use_cache=True):
    """Get Fama-French 5 factors at daily frequency."""
    if use_cache:
        cached = load_cache("ff5_daily")
        if cached is not None:
            return cached

    csv_text = _download_ff_csv("ff5_daily")
    df = _parse_ff5(csv_text, freq="daily")

    save_cache(df.reset_index(), "ff5_daily")
    return df


def get_momentum_monthly(use_cache=True):
    """Get momentum factor at monthly frequency."""
    if use_cache:
        cached = load_cache("mom_monthly")
        if cached is not None:
            return cached

    csv_text = _download_ff_csv("mom_monthly")
    df = _parse_momentum(csv_text, freq="monthly")

    save_cache(df.reset_index(), "mom_monthly")
    return df


def get_ff6_monthly(use_cache=True):
    """
    Get FF5 + momentum (6 factors) at monthly frequency.

    Returns
    -------
    DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    """
    ff5 = get_ff5_monthly(use_cache)
    mom = get_momentum_monthly(use_cache)

    if isinstance(ff5.index[0], str):
        ff5 = ff5.set_index("date")
    if isinstance(mom.index[0], str):
        mom = mom.set_index("date")

    merged = ff5.join(mom, how="inner")
    return merged
