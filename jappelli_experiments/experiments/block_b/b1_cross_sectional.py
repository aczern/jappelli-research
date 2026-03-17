"""
Experiment B1: Static Ownership and Cross-Sectional Returns

Tests Jappelli (2025) Proposition 2: cross-sectional return premium
from static fund ownership.

Key steps:
1. Compute stock-level static ownership SO_it from MF holdings
2. Fama-MacBeth: r_it = alpha + beta * SO_{it-1} + controls + eps
3. Portfolio sorts: quintiles on lagged SO, long-short spread, FF5 alpha
4. IV: instrument SO with S&P index inclusion, Russell reconstitution

Key outputs: Beta (ownership-return), portfolio spreads, alpha estimates
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from jappelli_experiments.config import (
    NW_LAGS_MONTHLY, ROLLING_BETA_WINDOW, WINSORIZE_PCTILE,
)
from jappelli_experiments.shared.fama_macbeth import fama_macbeth
from jappelli_experiments.shared.portfolio_sorts import single_sort, portfolio_alpha
from jappelli_experiments.shared.iv_regression import iv_2sls
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cleaning import winsorize_panel, lag_variable
from jappelli_experiments.data.cache import save_cache

logger = logging.getLogger(__name__)
setup_plots()


def compute_static_ownership(holdings, fund_class, crsp_msf):
    """
    Compute stock-level static fund ownership share SO_it.

    SO_it = (shares held by static funds) / (total shares outstanding)

    Parameters
    ----------
    holdings : DataFrame
        MF holdings with crsp_fundno, permno, report_dt, nbr_shares.
    fund_class : DataFrame
        Fund classification with crsp_fundno, is_static.
    crsp_msf : DataFrame
        CRSP monthly with permno, date, shrout.

    Returns
    -------
    DataFrame with permno, date, static_ownership, dynamic_ownership.
    """
    # Merge fund classification
    h = pd.merge(holdings, fund_class[["crsp_fundno", "is_static"]],
                  on="crsp_fundno", how="inner")

    # Aggregate shares by stock and fund type
    agg = h.groupby(["permno", "report_dt", "is_static"])["nbr_shares"].sum().reset_index()
    agg = agg.pivot_table(index=["permno", "report_dt"], columns="is_static",
                           values="nbr_shares", fill_value=0).reset_index()
    agg.columns.name = None

    if True in agg.columns:
        agg = agg.rename(columns={True: "static_shares", False: "dynamic_shares"})
    else:
        agg["static_shares"] = 0
        agg["dynamic_shares"] = agg.get(False, 0)

    # Merge with shares outstanding
    agg["date"] = pd.to_datetime(agg["report_dt"]) + pd.offsets.MonthEnd(0)

    msf_shrout = crsp_msf[["permno", "date", "shrout"]].copy()
    msf_shrout["date"] = pd.to_datetime(msf_shrout["date"]) + pd.offsets.MonthEnd(0)

    merged = pd.merge(agg, msf_shrout, on=["permno", "date"], how="left")

    # Compute ownership shares
    merged["shrout_total"] = merged["shrout"] * 1000  # shrout is in thousands
    merged["static_ownership"] = merged["static_shares"] / merged["shrout_total"]
    merged["dynamic_ownership"] = merged["dynamic_shares"] / merged["shrout_total"]
    merged["total_mf_ownership"] = (merged["static_shares"] + merged["dynamic_shares"]) / merged["shrout_total"]

    # Clip extreme values
    for col in ["static_ownership", "dynamic_ownership", "total_mf_ownership"]:
        merged[col] = merged[col].clip(0, 1)

    result = merged[["permno", "date", "static_ownership", "dynamic_ownership",
                      "total_mf_ownership"]].dropna()

    logger.info(f"Static ownership computed: {len(result):,} stock-quarters, "
                f"mean SO = {result['static_ownership'].mean():.4f}")

    return result


def run_fama_macbeth_tests(stock_panel, ff_factors=None):
    """
    Run Fama-MacBeth cross-sectional regressions of returns on static ownership.

    Parameters
    ----------
    stock_panel : DataFrame
        Stock-month panel with ret, static_ownership, and controls.
    ff_factors : DataFrame
        For computing risk-adjusted returns.

    Returns
    -------
    dict with FM results for multiple specifications.
    """
    df = stock_panel.copy()

    # Lag the sorting variable
    df["SO_lag"] = df.groupby("permno")["static_ownership"].shift(1)

    results = {}

    # Model 1: Univariate
    fm1 = fama_macbeth(df, "date", "ret", ["SO_lag"])
    results["univariate"] = fm1
    logger.info(f"  FM univariate: beta = {fm1.mean_coefs['SO_lag']:.4f} "
                f"(t = {fm1.t_stats['SO_lag']:.3f})")

    # Model 2: With size and value controls
    control_sets = {
        "size_value": ["log_me", "bm"],
        "full": ["log_me", "bm", "ret_lag"],
    }

    for name, controls in control_sets.items():
        available = [c for c in controls if c in df.columns]
        if not available:
            continue

        try:
            fm = fama_macbeth(df, "date", "ret", ["SO_lag"] + available)
            results[name] = fm
            logger.info(f"  FM {name}: beta = {fm.mean_coefs['SO_lag']:.4f} "
                        f"(t = {fm.t_stats['SO_lag']:.3f})")
        except Exception as e:
            logger.warning(f"  FM {name} failed: {e}")

    return results


def run_portfolio_sorts(stock_panel, ff_factors):
    """
    Portfolio sorts on static ownership.

    Parameters
    ----------
    stock_panel : DataFrame
        Stock-month panel.
    ff_factors : DataFrame
        FF5 factors for alpha computation.

    Returns
    -------
    dict with portfolio returns and alphas.
    """
    df = stock_panel.copy()
    df["SO_lag"] = df.groupby("permno")["static_ownership"].shift(1)

    # Equal-weighted quintile sorts
    port_rets_ew = single_sort(df, "date", "ret", "SO_lag", n_ports=5)

    # Value-weighted quintile sorts
    port_rets_vw = single_sort(df, "date", "ret", "SO_lag", n_ports=5,
                                weight_col="me")

    results = {
        "ew_returns": port_rets_ew,
        "vw_returns": port_rets_vw,
    }

    # Compute alphas for long-short portfolio
    if ff_factors is not None:
        factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                       if c in ff_factors.columns]
        if factor_cols and "LS" in port_rets_ew.columns:
            # EW alpha
            ew_alpha = portfolio_alpha(port_rets_ew["LS"], ff_factors, factor_cols)
            results["ew_ls_alpha"] = ew_alpha
            logger.info(f"  EW L/S alpha = {ew_alpha['alpha']*100:.2f}% "
                        f"(t = {ew_alpha['t_stat']:.3f})")

            # VW alpha
            vw_alpha = portfolio_alpha(port_rets_vw["LS"], ff_factors, factor_cols)
            results["vw_ls_alpha"] = vw_alpha
            logger.info(f"  VW L/S alpha = {vw_alpha['alpha']*100:.2f}% "
                        f"(t = {vw_alpha['t_stat']:.3f})")

    return results


def run_b1(stock_panel, holdings=None, fund_class=None, crsp_msf=None,
           ff_factors=None, sp500_events=None, save_outputs=True):
    """
    Run the full B1 experiment.

    Parameters
    ----------
    stock_panel : DataFrame
        Stock-month panel.
    holdings : DataFrame
        MF holdings data.
    fund_class : DataFrame
        Fund classification.
    crsp_msf : DataFrame
        CRSP monthly stock file.
    ff_factors : DataFrame
        Fama-French 5 factors.
    sp500_events : DataFrame
        S&P 500 add/drop for IV.
    save_outputs : bool
        Save results to disk.

    Returns
    -------
    dict with all B1 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT B1: Static Ownership & Cross-Sectional Returns")
    logger.info("=" * 60)

    results = {}
    mapper = ConnectionMapper()

    # --- Step 1: Compute static ownership ---
    if holdings is not None and fund_class is not None and crsp_msf is not None:
        logger.info("Step 1: Computing static ownership SO_it...")
        so = compute_static_ownership(holdings, fund_class, crsp_msf)
        stock_panel = pd.merge(stock_panel, so, on=["permno", "date"], how="left")
        results["so_stats"] = so.describe()
        mapper.register_output("B1", "SO_it", so[["permno", "date", "static_ownership"]])
    else:
        logger.warning("Missing data for SO computation; using existing panel columns")

    if "static_ownership" not in stock_panel.columns:
        logger.error("No static_ownership in panel. Cannot proceed.")
        return results

    # Prepare controls
    if "ret" in stock_panel.columns:
        stock_panel["ret_lag"] = stock_panel.groupby("permno")["ret"].shift(1)

    # --- Step 2: Fama-MacBeth ---
    logger.info("Step 2: Fama-MacBeth regressions...")
    fm_results = run_fama_macbeth_tests(stock_panel, ff_factors)
    results["fama_macbeth"] = fm_results

    # --- Step 3: Portfolio sorts ---
    logger.info("Step 3: Portfolio sorts...")
    if ff_factors is not None:
        port_results = run_portfolio_sorts(stock_panel, ff_factors)
        results["portfolio_sorts"] = port_results

    # --- Step 4: IV (if instruments available) ---
    if sp500_events is not None:
        logger.info("Step 4: IV regression...")
        # Construct instrument at stock level
        # (Stock was added to S&P 500 -> exogenous increase in static ownership)
        try:
            events = sp500_events.copy()
            if "permno" in events.columns and "date" in events.columns:
                events["added"] = 1
                events["date"] = pd.to_datetime(events["date"]) + pd.offsets.MonthEnd(0)
                panel_iv = pd.merge(stock_panel, events[["permno", "date", "added"]],
                                     on=["permno", "date"], how="left")
                panel_iv["added"] = panel_iv["added"].fillna(0)

                # Simple IV specification
                panel_iv["SO_lag"] = panel_iv.groupby("permno")["static_ownership"].shift(1)
                iv_data = panel_iv.dropna(subset=["ret", "SO_lag", "added"])

                if len(iv_data) > 100:
                    iv_result = iv_2sls(iv_data, "ret", ["SO_lag"], [], ["added"])
                    results["iv"] = iv_result
                    logger.info(f"  IV diagnostics: {iv_result['diagnostics']}")
        except Exception as e:
            logger.warning(f"  IV estimation failed: {e}")

    # --- Save outputs ---
    if save_outputs:
        # FM summary table
        fm_rows = []
        for name, fm in fm_results.items():
            summary = fm.summary()
            fm_rows.append({
                "Model": name,
                "SO_beta": summary.loc["SO_lag", "coef"],
                "SO_se": summary.loc["SO_lag", "se_nw"],
                "SO_t": summary.loc["SO_lag", "t_stat"],
                "SO_p": summary.loc["SO_lag", "p_value"],
                "N_periods": fm.n_periods,
                "Avg_N_stocks": f"{fm.avg_n_stocks:.0f}",
            })
        if fm_rows:
            to_csv(pd.DataFrame(fm_rows), "b1_fama_macbeth.csv")

        # Save SO_it for downstream
        if "static_ownership" in stock_panel.columns:
            save_cache(
                stock_panel[["permno", "date", "static_ownership"]].dropna(),
                "B1_SO_it"
            )

    logger.info("=" * 60)
    logger.info("B1 COMPLETE")
    logger.info("=" * 60)

    return results
