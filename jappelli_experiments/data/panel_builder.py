"""
Merged panel construction for aggregate, stock-level, and fund-level data.

Builds the three core panels used across all experiments:
1. Aggregate monthly panel (market index + theta + Sharpe + macro)
2. Stock-month panel (CRSP + Compustat + ownership + factor loadings)
3. Fund panel (TNA + flows + classification + allocation)
"""
import logging

import numpy as np
import pandas as pd

from jappelli_experiments.config import (
    STATIC_SD_THRESHOLD, MIN_FUND_TNA, ALLOC_BOUNDS,
    MF_SAMPLE_START, MF_SAMPLE_END, ROLLING_BETA_WINDOW,
)
from jappelli_experiments.data.cache import cached, save_cache
from jappelli_experiments.data.cleaning import winsorize, to_monthly

logger = logging.getLogger(__name__)


# ── Fund Classification ──

def classify_static_funds(mf_summary, sd_threshold=STATIC_SD_THRESHOLD,
                          min_tna=MIN_FUND_TNA, alloc_bounds=ALLOC_BOUNDS):
    """
    Classify mutual funds as static vs. dynamic following Jappelli (2025).

    Static funds: SD(equity allocation) <= threshold over full sample.

    Parameters
    ----------
    mf_summary : DataFrame
        CRSP mutual fund summary data with per_com (equity %) and mtna.
    sd_threshold : float
        Maximum SD of equity allocation to be classified as static.
    min_tna : float
        Minimum TNA in $M.
    alloc_bounds : tuple
        Valid allocation range (lower, upper) for filtering anomalies.

    Returns
    -------
    DataFrame with crsp_fundno and is_static indicator.
    """
    df = mf_summary.copy()

    # Filter: minimum TNA
    df = df[df["mtna"] >= min_tna]

    # Equity allocation (per_com is percent invested in common stock)
    df["eq_alloc"] = df["per_com"] / 100.0

    # Filter anomalous allocations
    df = df[(df["eq_alloc"] >= alloc_bounds[0] - 1) &
            (df["eq_alloc"] <= alloc_bounds[1])]

    # Compute SD of equity allocation per fund
    fund_stats = df.groupby("crsp_fundno").agg(
        eq_alloc_sd=("eq_alloc", "std"),
        eq_alloc_mean=("eq_alloc", "mean"),
        n_obs=("eq_alloc", "count"),
        avg_tna=("mtna", "mean"),
    ).reset_index()

    # Require minimum observations (at least 12 months)
    fund_stats = fund_stats[fund_stats["n_obs"] >= 12]

    # Classify
    fund_stats["is_static"] = fund_stats["eq_alloc_sd"] <= sd_threshold
    fund_stats["is_index"] = False  # Will be updated with index_fund_flag

    # Check for index fund flag if available
    if "index_fund_flag" in mf_summary.columns:
        idx_funds = mf_summary[mf_summary["index_fund_flag"].isin(["B", "D"])]["crsp_fundno"].unique()
        fund_stats.loc[fund_stats["crsp_fundno"].isin(idx_funds), "is_index"] = True

    logger.info(
        f"Fund classification: {fund_stats['is_static'].sum()} static / "
        f"{(~fund_stats['is_static']).sum()} dynamic out of {len(fund_stats)} funds"
    )

    return fund_stats


# ── Core Variable Construction ──

def compute_theta_t(mf_summary, fund_class):
    """
    Compute theta_t: aggregate static fund equity allocation / market cap.

    Parameters
    ----------
    mf_summary : DataFrame
        Fund summary with caldt, crsp_fundno, mtna, per_com.
    fund_class : DataFrame
        Fund classification with crsp_fundno, is_static.

    Returns
    -------
    Series with date index: theta_t values.
    """
    df = pd.merge(mf_summary, fund_class[["crsp_fundno", "is_static"]],
                   on="crsp_fundno", how="inner")

    # Static funds only
    static = df[df["is_static"]].copy()
    static["equity_exposure"] = static["mtna"] * static["per_com"] / 100.0

    # Aggregate equity exposure of static funds per month
    theta = static.groupby("caldt")["equity_exposure"].sum()
    theta.name = "theta_t"

    return theta


def compute_fund_flows(mf_summary):
    """
    Compute fund-level flows: TNA_t - TNA_{t-1} * (1 + r_t).

    Returns
    -------
    DataFrame with crsp_fundno, caldt, flow, flow_pct.
    """
    df = mf_summary[["crsp_fundno", "caldt", "mtna", "mret"]].copy()
    df = df.sort_values(["crsp_fundno", "caldt"])

    # Lag TNA
    df["tna_lag"] = df.groupby("crsp_fundno")["mtna"].shift(1)
    df["mret"] = df["mret"].fillna(0)

    # Flow = TNA_t - TNA_{t-1} * (1 + r_t)
    df["flow"] = df["mtna"] - df["tna_lag"] * (1 + df["mret"])
    df["flow_pct"] = df["flow"] / df["tna_lag"]

    # Winsorize extreme flows
    df["flow_pct"] = winsorize(df["flow_pct"], (0.01, 0.99))

    return df[["crsp_fundno", "caldt", "flow", "flow_pct", "mtna", "tna_lag"]].dropna()


def compute_aggregate_flows(fund_flows, fund_class):
    """
    Compute aggregate static and dynamic fund flows.

    Returns
    -------
    DataFrame with date, static_flow, dynamic_flow, total_flow.
    """
    df = pd.merge(fund_flows, fund_class[["crsp_fundno", "is_static"]],
                   on="crsp_fundno", how="inner")

    agg = df.groupby(["caldt", "is_static"])["flow"].sum().unstack("is_static")
    agg.columns = ["dynamic_flow", "static_flow"]
    agg["total_flow"] = agg["dynamic_flow"] + agg["static_flow"]

    return agg


# ── Panel Builders ──

@cached("aggregate_monthly_panel")
def build_aggregate_monthly_panel(crsp_msi=None, fred_data=None,
                                   ff_factors=None, theta=None,
                                   sharpe=None):
    """
    Build the aggregate monthly panel: market returns + theta + Sharpe + macro.

    Parameters
    ----------
    crsp_msi : DataFrame
        CRSP market index returns.
    fred_data : DataFrame
        FRED macro series.
    ff_factors : DataFrame
        Fama-French factors.
    theta : Series
        Aggregate theta_t.
    sharpe : Series
        Rolling Sharpe ratio.

    Returns
    -------
    DataFrame with monthly date index.
    """
    panels = []

    if crsp_msi is not None:
        msi = crsp_msi.set_index("date")[["vwretd", "ewretd", "totval"]].copy()
        msi.index = pd.to_datetime(msi.index) + pd.offsets.MonthEnd(0)
        panels.append(msi)

    if fred_data is not None:
        fred = fred_data.copy()
        if "date" in fred.columns:
            fred = fred.set_index("date")
        fred.index = pd.to_datetime(fred.index) + pd.offsets.MonthEnd(0)
        # Take last observation per month if not already monthly
        fred = fred.groupby(fred.index).last()
        panels.append(fred)

    if ff_factors is not None:
        ff = ff_factors.copy()
        if "date" in ff.columns:
            ff = ff.set_index("date")
        panels.append(ff)

    if theta is not None:
        t = theta.to_frame("theta_t")
        t.index = pd.to_datetime(t.index) + pd.offsets.MonthEnd(0)
        panels.append(t)

    if sharpe is not None:
        s = sharpe.to_frame("sharpe_t")
        s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
        panels.append(s)

    if not panels:
        raise ValueError("No data provided to build aggregate panel.")

    result = panels[0]
    for p in panels[1:]:
        result = result.join(p, how="outer")

    result = result.sort_index()
    logger.info(f"Aggregate monthly panel: {len(result)} months, {result.shape[1]} variables")
    return result


@cached("stock_month_panel")
def build_stock_month_panel(crsp_msf=None, compustat=None, ccm_link=None,
                             ff_factors=None):
    """
    Build stock-month panel: CRSP + Compustat + FF factor loadings.

    Returns
    -------
    DataFrame with (permno, date) as identifier.
    """
    if crsp_msf is None:
        raise ValueError("CRSP MSF data required.")

    df = crsp_msf.copy()

    # Merge Compustat via CCM link if available
    if compustat is not None and ccm_link is not None:
        # Match Compustat annual data to CRSP monthly
        comp = pd.merge(compustat, ccm_link[["gvkey", "permno", "linkdt", "linkenddt"]],
                         on="gvkey", how="inner")
        comp = comp[(comp["datadate"] >= comp["linkdt"]) &
                     (comp["datadate"] <= comp["linkenddt"])]

        # Lag Compustat by 6 months (availability delay)
        comp["merge_date"] = comp["datadate"] + pd.DateOffset(months=6)
        comp["merge_date"] = comp["merge_date"] + pd.offsets.MonthEnd(0)

        # Keep relevant columns
        comp_cols = ["permno", "merge_date", "be", "at", "sale", "ni"]
        comp_sub = comp[[c for c in comp_cols if c in comp.columns]].copy()
        comp_sub = comp_sub.rename(columns={"merge_date": "date"})

        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
        comp_sub["date"] = pd.to_datetime(comp_sub["date"]) + pd.offsets.MonthEnd(0)

        df = pd.merge(df, comp_sub, on=["permno", "date"], how="left")

        # Book-to-market
        if "be" in df.columns:
            df["bm"] = df["be"] / df["me"]
            df.loc[df["bm"] <= 0, "bm"] = np.nan

    # Log market equity
    df["log_me"] = np.log(df["me"].clip(lower=0.001))

    logger.info(f"Stock-month panel: {len(df):,} observations, "
                f"{df['permno'].nunique():,} stocks")
    return df


@cached("fund_panel")
def build_fund_panel(mf_summary=None, fund_class=None, fund_flows=None):
    """
    Build fund panel: TNA + flows + classification + allocation.

    Returns
    -------
    DataFrame with (crsp_fundno, caldt) as identifier.
    """
    if mf_summary is None:
        raise ValueError("MF summary data required.")

    df = mf_summary.copy()

    if fund_class is not None:
        df = pd.merge(df, fund_class[["crsp_fundno", "is_static", "is_index",
                                        "eq_alloc_sd", "eq_alloc_mean"]],
                       on="crsp_fundno", how="left")

    if fund_flows is not None:
        df = pd.merge(df, fund_flows[["crsp_fundno", "caldt", "flow", "flow_pct"]],
                       on=["crsp_fundno", "caldt"], how="left")

    logger.info(f"Fund panel: {len(df):,} fund-months, "
                f"{df['crsp_fundno'].nunique():,} funds")
    return df
