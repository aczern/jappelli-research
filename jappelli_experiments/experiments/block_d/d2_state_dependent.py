"""
Experiment D2: State-Dependent Elasticity and Spillovers

Tests whether flow elasticity varies with market conditions (VIX)
and whether equity static flows spill over into bond markets.

Key steps:
1. Cross-asset spillover: Delta_P_bonds on equity static flows
2. Within-equity: index vs non-index stock response
3. VIX interaction: flow_impact * High_VIX
4. Quantile regression on VIX quartiles

Key outputs: State-dependent elasticity, cross-asset spillover estimates
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from jappelli_experiments.config import NW_LAGS_MONTHLY
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig

logger = logging.getLogger(__name__)
setup_plots()


def cross_asset_spillover(panel, bond_ret_col, flow_col="static_flow_norm",
                           control_cols=None, nw_lags=NW_LAGS_MONTHLY):
    """
    Test whether equity static flows affect bond returns (spillover).

    r_bonds_t = alpha + beta * equity_static_flow_t + controls + eps

    Parameters
    ----------
    panel : DataFrame
    bond_ret_col : str
        Bond return column.
    flow_col : str
        Equity static flow column.

    Returns
    -------
    dict with spillover coefficient and diagnostics.
    """
    if control_cols is None:
        control_cols = []

    cols = [bond_ret_col, flow_col] + control_cols
    df = panel[cols].dropna()

    X = df[[flow_col] + control_cols].values
    y = df[bond_ret_col].values

    result = ols_nw(y, X, max_lag=nw_lags, add_constant=True)

    return {
        "beta_spillover": result.params[1],
        "se": result.bse[1],
        "t_stat": result.tvalues[1],
        "p_value": result.pvalues[1],
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
    }


def vix_interaction_test(panel, ret_col="vwretd", flow_col="static_flow_norm",
                          vix_col="vixcls", control_cols=None,
                          nw_lags=NW_LAGS_MONTHLY):
    """
    Test VIX-dependent flow elasticity via interaction term.

    r_t = alpha + beta_1 * flow + beta_2 * flow * High_VIX + controls + eps

    Parameters
    ----------
    vix_col : str
        VIX column.

    Returns
    -------
    dict with interaction results.
    """
    if control_cols is None:
        control_cols = []

    df = panel[[ret_col, flow_col, vix_col] + control_cols].dropna()

    # High VIX indicator (above median)
    df["high_vix"] = (df[vix_col] > df[vix_col].median()).astype(float)
    df["flow_x_vix"] = df[flow_col] * df["high_vix"]

    X_cols = [flow_col, "flow_x_vix", "high_vix"] + control_cols
    result = ols_nw(df[ret_col].values, df[X_cols].values,
                     max_lag=nw_lags, add_constant=True)

    return {
        "beta_base": result.params[1],
        "se_base": result.bse[1],
        "beta_interaction": result.params[2],
        "se_interaction": result.bse[2],
        "t_interaction": result.tvalues[2],
        "p_interaction": result.pvalues[2],
        "beta_low_vix": result.params[1],
        "beta_high_vix": result.params[1] + result.params[2],
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
    }


def vix_quartile_regressions(panel, ret_col="vwretd",
                              flow_col="static_flow_norm",
                              vix_col="vixcls", n_quantiles=4,
                              nw_lags=NW_LAGS_MONTHLY):
    """
    Estimate flow elasticity separately within VIX quartiles.

    Tests for monotonic increase in elasticity with VIX.

    Returns
    -------
    DataFrame with elasticity per VIX quartile.
    """
    df = panel[[ret_col, flow_col, vix_col]].dropna()
    df["vix_q"] = pd.qcut(df[vix_col], n_quantiles, labels=False) + 1

    results = []
    for q in range(1, n_quantiles + 1):
        sub = df[df["vix_q"] == q]
        if len(sub) < 20:
            continue

        res = ols_nw(sub[ret_col].values, sub[[flow_col]].values,
                      max_lag=nw_lags, add_constant=True)

        results.append({
            "vix_quartile": q,
            "mean_vix": sub[vix_col].mean(),
            "beta": res.params[1],
            "se": res.bse[1],
            "t_stat": res.tvalues[1],
            "p_value": res.pvalues[1],
            "n_obs": len(sub),
        })

    return pd.DataFrame(results)


def run_d2(aggregate_panel, save_outputs=True):
    """
    Run the full D2 experiment.

    Returns
    -------
    dict with all D2 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT D2: State-Dependent Elasticity & Spillovers")
    logger.info("=" * 60)

    results = {}
    panel = aggregate_panel.copy()

    # --- Cross-asset spillover ---
    bond_cols = [c for c in ["bamlc0a0cm", "dgs10"] if c in panel.columns]
    if bond_cols and "static_flow_norm" in panel.columns:
        logger.info("Step 1: Cross-asset spillover test...")
        for bc in bond_cols:
            # Convert yield change to "return" proxy
            panel[f"{bc}_change"] = -panel[bc].diff()  # Negative: yield up = price down
            spill = cross_asset_spillover(panel, f"{bc}_change")
            results[f"spillover_{bc}"] = spill
            logger.info(f"  {bc}: beta = {spill['beta_spillover']:.4f} "
                        f"(t = {spill['t_stat']:.3f})")

    # --- VIX interaction ---
    if "vixcls" in panel.columns and "static_flow_norm" in panel.columns:
        logger.info("Step 2: VIX interaction test...")
        vix_int = vix_interaction_test(panel)
        results["vix_interaction"] = vix_int
        logger.info(f"  Low VIX beta = {vix_int['beta_low_vix']:.4f}")
        logger.info(f"  High VIX beta = {vix_int['beta_high_vix']:.4f}")
        logger.info(f"  Interaction t = {vix_int['t_interaction']:.3f}")

        # --- VIX quartile regressions ---
        logger.info("Step 3: VIX quartile regressions...")
        vix_q = vix_quartile_regressions(panel)
        results["vix_quartiles"] = vix_q
        logger.info(f"\n{vix_q.to_string()}")

    # --- Save ---
    if save_outputs:
        if "vix_quartiles" in results:
            to_csv(results["vix_quartiles"], "d2_vix_quartile_elasticity.csv")

        if "vix_interaction" in results:
            to_csv(pd.DataFrame([results["vix_interaction"]]), "d2_vix_interaction.csv")

        # Figure: VIX-dependent elasticity
        if "vix_quartiles" in results:
            vq = results["vix_quartiles"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(vq["vix_quartile"], vq["beta"],
                   yerr=1.96 * vq["se"], capsize=5,
                   color=["#2ECC71", "#3498DB", "#F39C12", "#E74C3C"],
                   alpha=0.8)
            ax.set_xlabel("VIX Quartile (1=Low, 4=High)")
            ax.set_ylabel("Flow Elasticity (beta)")
            ax.set_title("State-Dependent Flow Elasticity by VIX Regime")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            save_fig(fig, "d2_vix_elasticity.pdf")

    logger.info("=" * 60)
    logger.info("D2 COMPLETE")
    logger.info("=" * 60)

    return results
