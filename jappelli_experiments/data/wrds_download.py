"""
WRDS data downloads for CRSP, Compustat, and Mutual Fund databases.

Adapted from Backus/code/wrds_downloads.py patterns.
Requires WRDS Python package and valid credentials.
"""
import logging

import numpy as np
import pandas as pd

from jappelli_experiments.data.cache import save_cache, load_cache

logger = logging.getLogger(__name__)


def _get_wrds_connection():
    """Establish WRDS database connection."""
    import wrds
    return wrds.Connection()


# ── CRSP Monthly Stock Index ──

def download_crsp_msi(use_cache=True):
    """
    Download CRSP Monthly Stock Index (aggregate market returns).

    Used by: A1 (aggregate returns)

    Returns
    -------
    DataFrame with date, vwretd, ewretd, totval, etc.
    """
    if use_cache:
        cached = load_cache("crsp_msi")
        if cached is not None:
            return cached

    db = _get_wrds_connection()
    query = """
        SELECT caldt as date, vwretd, vwretx, ewretd, ewretx,
               totval, totcnt, usdval, usdcnt
        FROM crsp.msi
        WHERE caldt >= '1926-01-01'
        ORDER BY caldt
    """
    df = db.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    db.close()

    save_cache(df, "crsp_msi")
    logger.info(f"Downloaded CRSP MSI: {len(df)} months")
    return df


# ── CRSP Monthly Stock File ──

def download_crsp_msf(use_cache=True, start_year=1980):
    """
    Download CRSP Monthly Stock File (full universe).

    Used by: B1, B2, E4

    Returns
    -------
    DataFrame with permno, date, ret, prc, shrout, vol, etc.
    """
    if use_cache:
        cached = load_cache("crsp_msf")
        if cached is not None:
            return cached

    db = _get_wrds_connection()
    query = f"""
        SELECT a.permno, a.permco, a.date, a.ret, a.retx, a.prc,
               a.altprc, a.shrout, a.vol, a.cfacshr, a.cfacpr,
               b.shrcd, b.exchcd, b.siccd, b.ticker, b.comnam
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt
        WHERE a.date >= '{start_year}-01-01'
            AND b.shrcd IN (10, 11)
            AND b.exchcd IN (1, 2, 3)
        ORDER BY a.permno, a.date
    """
    df = db.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)
    db.close()

    # Market equity
    df["me"] = df["prc"].abs() * df["shrout"] / 1000  # in $M

    save_cache(df, "crsp_msf")
    logger.info(f"Downloaded CRSP MSF: {len(df):,} stock-months")
    return df


# ── CRSP Mutual Fund Summary ──

def download_crsp_mf_summary(use_cache=True):
    """
    Download CRSP Mutual Fund Summary (TNA, returns, objectives).

    Used by: A1, A2, C1, D1, E3 (CRITICAL)

    Returns
    -------
    DataFrame with crsp_fundno, caldt, mtna, mret, fund_name, lipper_class, etc.
    """
    if use_cache:
        cached = load_cache("crsp_mf_summary")
        if cached is not None:
            return cached

    db = _get_wrds_connection()
    query = """
        SELECT a.crsp_fundno, a.caldt, a.mtna, a.mret, a.mnav,
               b.fund_name, b.lipper_class, b.lipper_obj_cd,
               b.si_obj_cd, b.wbrger_obj_cd,
               b.per_com, b.per_pref, b.per_conv, b.per_corp,
               b.per_muni, b.per_govt, b.per_oth, b.per_cash,
               b.per_bond, b.per_abs, b.per_mbs, b.per_eq_oth,
               b.per_fi_oth, b.lipper_asset_cd, b.lipper_class_name,
               b.index_fund_flag, b.et_flag
        FROM crsp.monthly_tna_ret_nav AS a
        LEFT JOIN crsp.fund_summary AS b
            ON a.crsp_fundno = b.crsp_fundno
            AND a.caldt >= b.begdt
            AND a.caldt <= b.enddt
        WHERE a.caldt >= '2000-01-01'
        ORDER BY a.crsp_fundno, a.caldt
    """
    df = db.raw_sql(query)
    df["caldt"] = pd.to_datetime(df["caldt"])
    db.close()

    save_cache(df, "crsp_mf_summary")
    logger.info(f"Downloaded CRSP MF Summary: {len(df):,} fund-months")
    return df


# ── CRSP Mutual Fund Holdings ──

def download_crsp_mf_holdings(use_cache=True, start_year=2004):
    """
    Download CRSP Mutual Fund Holdings.

    Used by: B1, B2, D1, E3 (CRITICAL)

    Note: This is a large dataset (~10GB). Downloads in yearly chunks.

    Returns
    -------
    DataFrame with crsp_fundno, report_dt, security_name, permno, nbr_shares, etc.
    """
    if use_cache:
        cached = load_cache("crsp_mf_holdings")
        if cached is not None:
            return cached

    db = _get_wrds_connection()
    frames = []

    for year in range(start_year, 2025):
        logger.info(f"Downloading MF holdings for {year}...")
        query = f"""
            SELECT crsp_fundno, report_dt, security_name,
                   cusip, permno, permco,
                   nbr_shares, market_val, pct_tna,
                   crsp_portno, crsp_company_key
            FROM crsp.holdings
            WHERE report_dt >= '{year}-01-01'
              AND report_dt < '{year + 1}-01-01'
            ORDER BY crsp_fundno, report_dt
        """
        chunk = db.raw_sql(query)
        if len(chunk) > 0:
            chunk["report_dt"] = pd.to_datetime(chunk["report_dt"])
            frames.append(chunk)

    db.close()

    if frames:
        df = pd.concat(frames, ignore_index=True)
        save_cache(df, "crsp_mf_holdings")
        logger.info(f"Downloaded CRSP MF Holdings: {len(df):,} fund-stock-quarters")
        return df

    return pd.DataFrame()


# ── Compustat Annual Fundamentals ──

def download_compustat_funda(use_cache=True):
    """
    Download Compustat Annual Fundamentals.

    Used by: B1, B2 (firm characteristics)

    Returns
    -------
    DataFrame with gvkey, datadate, at, ceq, sale, ni, etc.
    """
    if use_cache:
        cached = load_cache("compustat_funda")
        if cached is not None:
            return cached

    db = _get_wrds_connection()
    query = """
        SELECT gvkey, datadate, fyear,
               at, ceq, seq, txditc, pstkrv, pstkl, pstk,
               lt, dltt, dlc, che, ivao,
               sale, revt, cogs, xsga, xrd, dp, xint, ni,
               oibdp, ebitda, ib,
               csho, prcc_f, dvt, dvc,
               sich, naicsh
        FROM comp.funda
        WHERE fyear >= 1978
            AND indfmt = 'INDL'
            AND datafmt = 'STD'
            AND popsrc = 'D'
            AND consol = 'C'
        ORDER BY gvkey, datadate
    """
    df = db.raw_sql(query)
    df["datadate"] = pd.to_datetime(df["datadate"])
    db.close()

    # Book equity
    df["be"] = df["ceq"] + df["txditc"].fillna(0) - df["pstkrv"].fillna(
        df["pstkl"].fillna(df["pstk"].fillna(0))
    )

    save_cache(df, "compustat_funda")
    logger.info(f"Downloaded Compustat: {len(df):,} firm-years")
    return df


# ── CRSP-Compustat Link ──

def download_ccm_link(use_cache=True):
    """Download CRSP-Compustat Merged link table."""
    if use_cache:
        cached = load_cache("ccm_link")
        if cached is not None:
            return cached

    db = _get_wrds_connection()
    query = """
        SELECT gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU', 'LC')
            AND linkprim IN ('P', 'C')
    """
    df = db.raw_sql(query)
    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])
    df.loc[df["linkenddt"].isna(), "linkenddt"] = pd.Timestamp("2099-12-31")
    df["permno"] = df["permno"].astype(int)
    db.close()

    save_cache(df, "ccm_link")
    return df
