"""
WRDS data downloads for CRSP, Compustat, and Mutual Fund databases.

Adapted from Backus/code/wrds_downloads.py patterns.
Requires WRDS Python package and valid credentials.

Schema verified against WRDS as of 2026-03.
"""
import logging

import numpy as np
import pandas as pd

from jappelli_experiments.data.cache import save_cache, load_cache, CACHE_DIR

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
    """
    if use_cache:
        cached = load_cache("crsp_msi")
        if cached is not None:
            return cached

    db = _get_wrds_connection()
    query = """
        SELECT date, vwretd, vwretx, ewretd, ewretx,
               sprtrn, spindx, totval, totcnt, usdval, usdcnt
        FROM crsp.msi
        WHERE date >= '1926-01-01'
        ORDER BY date
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
# Schema: monthly_tna_ret_nav has (caldt, crsp_fundno, mtna, mret, mnav)
#         fund_summary has (caldt, crsp_fundno, summary_period, per_com, ...)
#         fund_style has (crsp_fundno, begdt, enddt, lipper_class, ...)
#         fund_hdr has (crsp_fundno, fund_name, index_fund_flag, et_flag, ...)

def download_crsp_mf_summary(use_cache=True):
    """
    Download CRSP Mutual Fund Summary: TNA, returns, allocations, and classifications.

    Joins:
      - monthly_tna_ret_nav (TNA, returns)
      - fund_summary (asset allocation percentages)
      - fund_style (lipper classification, date-range join)
      - fund_hdr (fund name, index flag, one row per fund)

    Used by: A1, A2, C1, D1, E3 (CRITICAL)
    """
    if use_cache:
        cached = load_cache("crsp_mf_summary")
        if cached is not None:
            return cached

    db = _get_wrds_connection()

    # Step 1: TNA and returns (monthly, ~10M rows)
    logger.info("  Downloading monthly_tna_ret_nav...")
    q_tna = """
        SELECT crsp_fundno, caldt, mtna, mret, mnav
        FROM crsp.monthly_tna_ret_nav
        WHERE caldt >= '2000-01-01'
        ORDER BY crsp_fundno, caldt
    """
    df_tna = db.raw_sql(q_tna)
    df_tna["caldt"] = pd.to_datetime(df_tna["caldt"])
    df_tna["crsp_fundno"] = df_tna["crsp_fundno"].astype(int)
    logger.info(f"    {len(df_tna):,} fund-months")

    # Step 2: Asset allocations from fund_summary (~3M rows, may have
    # multiple summary_period values per caldt — keep only the most
    # granular/monthly and dedup)
    logger.info("  Downloading fund_summary (allocations)...")
    q_alloc = """
        SELECT crsp_fundno, caldt,
               per_com, per_pref, per_conv, per_corp,
               per_muni, per_govt, per_oth, per_cash,
               per_bond, per_abs, per_mbs, per_eq_oth, per_fi_oth
        FROM crsp.fund_summary
        WHERE caldt >= '2000-01-01'
        ORDER BY crsp_fundno, caldt
    """
    df_alloc = db.raw_sql(q_alloc)
    df_alloc["caldt"] = pd.to_datetime(df_alloc["caldt"])
    df_alloc["crsp_fundno"] = df_alloc["crsp_fundno"].astype(int)
    # Dedup: keep last row per (crsp_fundno, caldt) in case of
    # multiple summary periods
    df_alloc = df_alloc.drop_duplicates(
        subset=["crsp_fundno", "caldt"], keep="last"
    )
    logger.info(f"    {len(df_alloc):,} rows (after dedup)")

    # Step 3: Fund style / classification (~200K rows, date-range keyed)
    logger.info("  Downloading fund_style (classifications)...")
    q_style = """
        SELECT crsp_fundno, begdt, enddt,
               crsp_obj_cd, si_obj_cd, wbrger_obj_cd,
               lipper_class, lipper_class_name, lipper_obj_cd,
               lipper_asset_cd
        FROM crsp.fund_style
    """
    df_style = db.raw_sql(q_style)
    df_style["begdt"] = pd.to_datetime(df_style["begdt"])
    df_style["enddt"] = pd.to_datetime(df_style["enddt"])
    df_style["crsp_fundno"] = df_style["crsp_fundno"].astype(int)
    logger.info(f"    {len(df_style):,} rows")

    # Step 4: Fund header — one row per fund (name, index flag)
    logger.info("  Downloading fund_hdr (names, flags)...")
    q_hdr = """
        SELECT crsp_fundno, crsp_portno, fund_name,
               index_fund_flag, et_flag
        FROM crsp.fund_hdr
    """
    df_hdr = db.raw_sql(q_hdr)
    df_hdr["crsp_fundno"] = df_hdr["crsp_fundno"].astype(int)
    # Dedup in case of multiple rows per fundno
    df_hdr = df_hdr.drop_duplicates(subset=["crsp_fundno"], keep="last")
    logger.info(f"    {len(df_hdr):,} funds (after dedup)")

    db.close()

    # ── Merge everything ──
    logger.info("  Merging TNA + allocations...")
    df = pd.merge(df_tna, df_alloc, on=["crsp_fundno", "caldt"], how="left")
    # per_com will be NaN for months without a fund_summary row;
    # forward-fill within each fund so quarterly allocations cover
    # the intervening months
    alloc_cols = [c for c in df_alloc.columns if c.startswith("per_")]
    df[alloc_cols] = df.groupby("crsp_fundno")[alloc_cols].ffill()
    logger.info(f"    {len(df):,} rows after TNA+alloc merge")

    # Add style (date-range join — careful to keep rows without style)
    logger.info("  Merging style classifications...")
    df = pd.merge(df, df_style, on="crsp_fundno", how="left")
    # Keep rows that fall within the style date range OR have no style info
    in_range = (df["caldt"] >= df["begdt"]) & (df["caldt"] <= df["enddt"])
    no_style = df["begdt"].isna()
    df = df[in_range | no_style].copy()
    df = df.drop(columns=["begdt", "enddt"])
    # Drop duplicates from overlapping style periods
    df = df.drop_duplicates(subset=["crsp_fundno", "caldt"], keep="last")
    logger.info(f"    {len(df):,} rows after style merge")

    # Add header info (one row per fund — no expansion)
    logger.info("  Merging fund header...")
    df = pd.merge(df, df_hdr, on="crsp_fundno", how="left")

    df = df.sort_values(["crsp_fundno", "caldt"]).reset_index(drop=True)

    save_cache(df, "crsp_mf_summary")
    logger.info(f"Downloaded CRSP MF Summary: {len(df):,} fund-months, "
                f"{df['crsp_fundno'].nunique():,} funds")
    return df


# ── CRSP Mutual Fund Holdings ──
# Schema: holdings has crsp_portno (NOT crsp_fundno), ~438M rows total
#         crsp_portno_map links crsp_fundno <-> crsp_portno (date range)
#         Column is percent_tna (NOT pct_tna)
#
# Strategy: download and save each year as a separate parquet to avoid OOM.
# Provide load_holdings_year() for lazy access.

def download_crsp_mf_holdings(use_cache=True, start_year=2004):
    """
    Download CRSP Mutual Fund Holdings, saved per-year as Parquet.

    Holdings are keyed on crsp_portno. We join with crsp_portno_map
    to get crsp_fundno for linking to fund-level data.

    Each year is saved as a separate file:
        cache/crsp_mf_holdings_{year}.parquet

    Used by: B1, B2, D1, E3 (CRITICAL)

    Returns
    -------
    DataFrame for the LAST year downloaded (as a sanity check).
    Full data should be accessed via load_holdings_year(year).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if all years are already cached
    if use_cache:
        all_cached = all(
            (CACHE_DIR / f"crsp_mf_holdings_{y}.parquet").exists()
            for y in range(start_year, 2025)
        )
        if all_cached:
            logger.info("All holdings years already cached.")
            # Return last year as sample
            return pd.read_parquet(CACHE_DIR / f"crsp_mf_holdings_2024.parquet")

    db = _get_wrds_connection()

    # First: get the portno -> fundno mapping
    logger.info("  Downloading portno map...")
    q_map = """
        SELECT crsp_fundno, crsp_portno, begdt, enddt
        FROM crsp.crsp_portno_map
    """
    portno_map = db.raw_sql(q_map)
    portno_map["crsp_fundno"] = portno_map["crsp_fundno"].astype(int)
    portno_map["crsp_portno"] = portno_map["crsp_portno"].astype(int)
    portno_map["begdt"] = pd.to_datetime(portno_map["begdt"])
    portno_map["enddt"] = pd.to_datetime(portno_map["enddt"])
    logger.info(f"    {len(portno_map):,} portno mappings")

    total_rows = 0
    last_chunk = None

    for year in range(start_year, 2025):
        out_path = CACHE_DIR / f"crsp_mf_holdings_{year}.parquet"

        # Skip if already cached
        if use_cache and out_path.exists():
            logger.info(f"  {year}: already cached, skipping")
            continue

        logger.info(f"  Downloading holdings for {year}...")
        query = f"""
            SELECT crsp_portno, report_dt, security_name,
                   cusip, permno, permco,
                   nbr_shares, market_val, percent_tna,
                   crsp_company_key
            FROM crsp.holdings
            WHERE report_dt >= '{year}-01-01'
              AND report_dt < '{year + 1}-01-01'
            ORDER BY crsp_portno, report_dt
        """
        chunk = db.raw_sql(query)

        if len(chunk) == 0:
            logger.info(f"    {year}: no data")
            continue

        chunk["report_dt"] = pd.to_datetime(chunk["report_dt"])
        chunk["crsp_portno"] = chunk["crsp_portno"].astype(int)

        # Map crsp_portno -> crsp_fundno for this year's data
        chunk = pd.merge(chunk, portno_map, on="crsp_portno", how="left")
        chunk = chunk[
            (chunk["report_dt"] >= chunk["begdt"])
            & (chunk["report_dt"] <= chunk["enddt"])
        ].copy()
        chunk = chunk.drop(columns=["begdt", "enddt"])

        # Rename for consistency
        chunk = chunk.rename(columns={"percent_tna": "pct_tna"})

        # Save this year
        chunk.to_parquet(out_path, index=False)
        total_rows += len(chunk)
        last_chunk = chunk
        logger.info(f"    {year}: {len(chunk):,} rows saved")

    db.close()

    logger.info(f"Downloaded CRSP MF Holdings: {total_rows:,} total rows "
                f"across {2025 - start_year} years")

    return last_chunk if last_chunk is not None else pd.DataFrame()


def load_holdings_year(year):
    """
    Load a single year of holdings data from cache.

    Parameters
    ----------
    year : int

    Returns
    -------
    DataFrame
    """
    path = CACHE_DIR / f"crsp_mf_holdings_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Holdings for {year} not cached. Run download_crsp_mf_holdings() first."
        )
    return pd.read_parquet(path)


def load_holdings_range(start_year, end_year):
    """
    Load multiple years of holdings and concatenate.

    Only use this if you have enough RAM for the requested range.
    """
    frames = []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(load_holdings_year(year))
        except FileNotFoundError:
            logger.warning(f"Holdings for {year} not found, skipping")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ── Compustat Annual Fundamentals ──

def download_compustat_funda(use_cache=True):
    """
    Download Compustat Annual Fundamentals.

    Used by: B1, B2 (firm characteristics)
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
