"""
Experiment D1: Static vs. Dynamic Flow Elasticity

Decomposes the aggregate flow-return elasticity into static and dynamic
components to test whether static flows have disproportionate price impact.

Key steps:
1. Classify funds: static (SD equity alloc <= 5%) vs dynamic
2. Separate flow series: Delta_F_static/MV and Delta_F_dynamic/MV
3. Horse-race regression: Delta_P = beta_static * static + beta_dynamic * dynamic
4. IV both flow types using different instrument sets
5. Test H0: beta_static = beta_dynamic
6. Benchmark against Haddad demand-system elasticities

Key outputs: Static vs dynamic elasticity, equality test
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from jappelli_experiments.config import NW_LAGS_MONTHLY
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.iv_regression import iv_2sls
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cache import save_cache

logger = logging.getLogger(__name__)
setup_plots()


def horse_race_regression(panel, ret_col="vwretd",
                           static_col="static_flow_norm",
                           dynamic_col="dynamic_flow_norm",
                           control_cols=None, nw_lags=NW_LAGS_MONTHLY):
    """
    Horse-race regression: returns on static and dynamic flows simultaneously.

    r_t = alpha + beta_s * static_flow + beta_d * dynamic_flow + controls + eps

    Parameters
    ----------
    panel : DataFrame
    ret_col : str
    static_col : str
    dynamic_col : str
    control_cols : list of str or None
    nw_lags : int

    Returns
    -------
    dict with coefficients, equality test, and diagnostics.
    """
    if control_cols is None:
        control_cols = []

    cols = [ret_col, static_col, dynamic_col] + control_cols
    df = panel[cols].dropna()

    X_cols = [static_col, dynamic_col] + control_cols
    result = ols_nw(df[ret_col].values, df[X_cols].values,
                     max_lag=nw_lags, add_constant=True)

    beta_s = result.params[1]
    beta_d = result.params[2]
    se_s = result.bse[1]
    se_d = result.bse[2]

    # Test H0: beta_static = beta_dynamic
    # Wald test: (beta_s - beta_d) / se(beta_s - beta_d)
    # Need covariance between coefficients
    cov_matrix = result.cov_params()
    var_diff = cov_matrix[1, 1] + cov_matrix[2, 2] - 2 * cov_matrix[1, 2]
    se_diff = np.sqrt(max(var_diff, 0))
    t_diff = (beta_s - beta_d) / se_diff if se_diff > 0 else np.nan
    p_diff = 2 * (1 - sp_stats.t.cdf(abs(t_diff), df=result.df_resid))

    return {
        "beta_static": beta_s,
        "se_static": se_s,
        "t_static": result.tvalues[1],
        "p_static": result.pvalues[1],
        "beta_dynamic": beta_d,
        "se_dynamic": se_d,
        "t_dynamic": result.tvalues[2],
        "p_dynamic": result.pvalues[2],
        "ratio": beta_s / beta_d if beta_d != 0 else np.inf,
        "diff": beta_s - beta_d,
        "se_diff": se_diff,
        "t_diff": t_diff,
        "p_diff": p_diff,
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
        "full_results": result,
    }


def haddad_benchmark_comparison(our_elasticity, haddad_elasticities):
    """
    Compare our reduced-form elasticity estimates to Haddad demand-system.

    Parameters
    ----------
    our_elasticity : dict
        Our estimated elasticities (static and dynamic).
    haddad_elasticities : DataFrame
        Haddad Ek estimates.

    Returns
    -------
    DataFrame comparing estimates.
    """
    comparison = pd.DataFrame([{
        "Source": "This paper (reduced-form)",
        "Static elasticity": our_elasticity.get("beta_static", np.nan),
        "Dynamic elasticity": our_elasticity.get("beta_dynamic", np.nan),
        "Ratio (static/dynamic)": our_elasticity.get("ratio", np.nan),
    }])

    if haddad_elasticities is not None and len(haddad_elasticities) > 0:
        # Average Haddad elasticities across quarters
        if "Ek" in haddad_elasticities.columns:
            haddad_mean = haddad_elasticities["Ek"].mean()
            comparison = pd.concat([comparison, pd.DataFrame([{
                "Source": "Haddad et al. (demand system)",
                "Static elasticity": np.nan,
                "Dynamic elasticity": np.nan,
                "Ratio (static/dynamic)": np.nan,
                "Demand elasticity (avg)": haddad_mean,
            }])], ignore_index=True)

    return comparison


def run_d1(aggregate_panel, fund_flows=None, fund_class=None,
           haddad_data=None, save_outputs=True):
    """
    Run the full D1 experiment.

    Parameters
    ----------
    aggregate_panel : DataFrame
        Aggregate panel with returns.
    fund_flows : DataFrame
        Fund-level flows.
    fund_class : DataFrame
        Fund classification.
    haddad_data : DataFrame
        Haddad elasticity estimates for benchmarking.
    save_outputs : bool
        Save results.

    Returns
    -------
    dict with all D1 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT D1: Static vs. Dynamic Flow Elasticity")
    logger.info("=" * 60)

    results = {}
    mapper = ConnectionMapper()

    panel = aggregate_panel.copy()

    # --- Horse-race regression ---
    if "static_flow_norm" in panel.columns and "dynamic_flow_norm" in panel.columns:
        logger.info("Step 1: Horse-race regression...")

        # Without controls
        hr = horse_race_regression(panel)
        results["horse_race"] = hr
        logger.info(f"  beta_static = {hr['beta_static']:.4f} (t = {hr['t_static']:.3f})")
        logger.info(f"  beta_dynamic = {hr['beta_dynamic']:.4f} (t = {hr['t_dynamic']:.3f})")
        logger.info(f"  Ratio (static/dynamic) = {hr['ratio']:.2f}")
        logger.info(f"  Equality test: t = {hr['t_diff']:.3f}, p = {hr['p_diff']:.3f}")

        # With controls
        controls = [c for c in ["dgs10", "vixcls"] if c in panel.columns]
        if controls:
            hr_ctrl = horse_race_regression(panel, control_cols=controls)
            results["horse_race_controls"] = hr_ctrl
            logger.info(f"  With controls: ratio = {hr_ctrl['ratio']:.2f}")

    # --- Haddad benchmark ---
    if haddad_data is not None and "horse_race" in results:
        logger.info("Step 2: Haddad demand-system benchmark comparison...")
        comparison = haddad_benchmark_comparison(results["horse_race"], haddad_data)
        results["haddad_comparison"] = comparison
        logger.info(f"\n{comparison.to_string()}")

    # --- Save ---
    if save_outputs:
        if "horse_race" in results:
            hr = results["horse_race"]
            table = pd.DataFrame([{
                "Variable": "Static flows",
                "beta": hr["beta_static"],
                "SE": hr["se_static"],
                "t": hr["t_static"],
                "p": hr["p_static"],
            }, {
                "Variable": "Dynamic flows",
                "beta": hr["beta_dynamic"],
                "SE": hr["se_dynamic"],
                "t": hr["t_dynamic"],
                "p": hr["p_dynamic"],
            }, {
                "Variable": "Difference",
                "beta": hr["diff"],
                "SE": hr["se_diff"],
                "t": hr["t_diff"],
                "p": hr["p_diff"],
            }])
            to_csv(table, "d1_flow_elasticity_decomposition.csv")

        # Figure: elasticity comparison
        if "horse_race" in results:
            fig, ax = plt.subplots(figsize=(8, 5))
            hr = results["horse_race"]
            categories = ["Static", "Dynamic"]
            betas = [hr["beta_static"], hr["beta_dynamic"]]
            ses = [hr["se_static"], hr["se_dynamic"]]
            colors = ["#2C3E50", "#E74C3C"]

            bars = ax.bar(categories, betas, yerr=[1.96 * s for s in ses],
                          capsize=5, color=colors, alpha=0.8)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Flow Elasticity (beta)")
            ax.set_title("Static vs. Dynamic Flow Elasticity")
            ax.text(0.5, 0.95, f"Ratio: {hr['ratio']:.2f}x\np(equality) = {hr['p_diff']:.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            save_fig(fig, "d1_elasticity_comparison.pdf")

    logger.info("=" * 60)
    logger.info("D1 COMPLETE")
    logger.info("=" * 60)

    return results
