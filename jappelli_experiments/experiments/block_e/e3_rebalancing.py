"""
Experiment E3: Portfolio Rebalancing Behavior

Classifies static funds by rebalancing approach (calendar, threshold,
constant proportion) and estimates separate elasticities by type.

Key steps:
1. Classify funds by observed rebalancing patterns from holdings data
2. Separate elasticity by rebalancing type
3. Calendar exogeneity: use month dummies as instruments

Key outputs: Rebalancing classification, type-specific elasticities
"""
import logging

import numpy as np
import pandas as pd

from jappelli_experiments.config import NW_LAGS_MONTHLY
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig

logger = logging.getLogger(__name__)
setup_plots()


def classify_rebalancing_pattern(holdings, fund_class):
    """
    Classify fund rebalancing behavior from holdings patterns.

    Categories:
    - Calendar: rebalances at fixed intervals (quarterly, semi-annual)
    - Threshold: rebalances when allocation drifts beyond band
    - Constant proportion: maintains near-constant weights

    Parameters
    ----------
    holdings : DataFrame
        Fund holdings over time.
    fund_class : DataFrame
        Fund classification (static funds only).

    Returns
    -------
    DataFrame with crsp_fundno, rebalancing_type, characteristics.
    """
    static_funds = fund_class[fund_class["is_static"]]["crsp_fundno"].unique()
    h = holdings[holdings["crsp_fundno"].isin(static_funds)].copy()

    results = []

    for fundno in h["crsp_fundno"].unique():
        fund_h = h[h["crsp_fundno"] == fundno].sort_values("report_dt")

        if len(fund_h) < 8:  # Need at least 2 years
            continue

        # Compute portfolio weights over time
        if "pct_tna" in fund_h.columns:
            weights = fund_h.groupby("report_dt")["pct_tna"].std()
        else:
            continue

        # Classify based on trading patterns
        n_changes = (weights.diff().abs() > 0.01).sum()
        n_periods = len(weights)
        change_rate = n_changes / n_periods if n_periods > 0 else 0

        # Calendar pattern: regular intervals between rebalances
        change_dates = weights.index[weights.diff().abs() > 0.01]
        if len(change_dates) >= 3:
            intervals = np.diff([d.toordinal() for d in change_dates])
            interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 999
        else:
            interval_cv = 999

        if interval_cv < 0.3 and change_rate > 0.2:
            reb_type = "calendar"
        elif change_rate < 0.1:
            reb_type = "constant_proportion"
        else:
            reb_type = "threshold"

        results.append({
            "crsp_fundno": fundno,
            "rebalancing_type": reb_type,
            "change_rate": change_rate,
            "interval_cv": interval_cv,
            "n_observations": n_periods,
        })

    df = pd.DataFrame(results)
    logger.info(f"Rebalancing classification: "
                f"{(df['rebalancing_type']=='calendar').sum()} calendar, "
                f"{(df['rebalancing_type']=='threshold').sum()} threshold, "
                f"{(df['rebalancing_type']=='constant_proportion').sum()} constant")
    return df


def elasticity_by_rebalancing_type(panel, reb_class, fund_flows,
                                     nw_lags=NW_LAGS_MONTHLY):
    """
    Estimate flow elasticity separately by rebalancing type.

    Returns
    -------
    DataFrame with elasticity per rebalancing type.
    """
    results = []

    for reb_type in ["calendar", "threshold", "constant_proportion"]:
        funds = reb_class[reb_class["rebalancing_type"] == reb_type]["crsp_fundno"].unique()

        # Aggregate flows for this type
        type_flows = fund_flows[fund_flows["crsp_fundno"].isin(funds)]
        agg_flow = type_flows.groupby("caldt")["flow"].sum()

        if len(agg_flow) < 30:
            continue

        # Merge with returns
        df = pd.merge(
            panel[["vwretd"]],
            agg_flow.rename("flow"),
            left_index=True, right_index=True, how="inner"
        ).dropna()

        if len(df) < 30:
            continue

        result = ols_nw(df["vwretd"].values, df[["flow"]].values,
                         max_lag=nw_lags, add_constant=True)

        results.append({
            "rebalancing_type": reb_type,
            "n_funds": len(funds),
            "beta": result.params[1],
            "se": result.bse[1],
            "t_stat": result.tvalues[1],
            "p_value": result.pvalues[1],
            "n_obs": int(result.nobs),
        })

    return pd.DataFrame(results)


def run_e3(aggregate_panel, holdings=None, fund_class=None,
           fund_flows=None, save_outputs=True):
    """
    Run the full E3 experiment.

    Returns
    -------
    dict with all E3 results.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT E3: Portfolio Rebalancing Behavior")
    logger.info("=" * 60)

    results = {}

    if holdings is None or fund_class is None:
        logger.warning("Holdings or fund classification not available. "
                       "E3 requires detailed holdings data.")
        return results

    # --- Classify rebalancing ---
    logger.info("Step 1: Classifying rebalancing patterns...")
    reb_class = classify_rebalancing_pattern(holdings, fund_class)
    results["classification"] = reb_class

    # --- Elasticity by type ---
    if fund_flows is not None:
        logger.info("Step 2: Estimating elasticity by rebalancing type...")
        elas = elasticity_by_rebalancing_type(aggregate_panel, reb_class, fund_flows)
        results["elasticity_by_type"] = elas
        logger.info(f"\n{elas.to_string()}")

    # --- Save ---
    if save_outputs:
        to_csv(reb_class, "e3_rebalancing_classification.csv")
        if "elasticity_by_type" in results:
            to_csv(results["elasticity_by_type"], "e3_elasticity_by_type.csv")

    logger.info("=" * 60)
    logger.info("E3 COMPLETE")
    logger.info("=" * 60)

    return results
