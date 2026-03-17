"""
Experiment E4: Factor-Level Bubbles

Tests whether static fund ownership creates factor-level price pressure,
extending the bubble framework to systematic risk factors.

Key steps:
1. Compute static fund factor exposures: sum(SO_i * beta_i^Factor) across stocks
2. Factor-level price pressure: Factor_Return ~ Static_Exp^Factor
3. VAR extension with multiple factors
4. Sector-level double sorts

Key outputs: Factor-level bubble estimates, factor-return predictions
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from jappelli_experiments.config import NW_LAGS_MONTHLY, ROLLING_BETA_WINDOW
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.var_models import estimate_var, impulse_responses
from jappelli_experiments.shared.fama_macbeth import fama_macbeth
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cache import load_cache

logger = logging.getLogger(__name__)
setup_plots()


def compute_static_factor_exposure(so_it, factor_loadings, factors):
    """
    Compute aggregate static fund exposure to each factor.

    Static_Exp_f,t = sum_i (SO_i,t * beta_i,f,t) for each factor f

    Parameters
    ----------
    so_it : DataFrame
        Stock-level static ownership with permno, date, static_ownership.
    factor_loadings : DataFrame
        Stock-level factor loadings with permno, date, and factor beta columns.
    factors : list of str
        Factor names (matching columns in factor_loadings).

    Returns
    -------
    DataFrame with date index and one column per factor exposure.
    """
    merged = pd.merge(so_it, factor_loadings, on=["permno", "date"], how="inner")

    exposures = {}
    for factor in factors:
        if factor in merged.columns:
            merged[f"weighted_{factor}"] = merged["static_ownership"] * merged[factor]
            exp = merged.groupby("date")[f"weighted_{factor}"].sum()
            exposures[f"static_exp_{factor}"] = exp

    return pd.DataFrame(exposures)


def factor_pressure_regressions(factor_returns, static_exposures,
                                  nw_lags=NW_LAGS_MONTHLY):
    """
    Regress factor returns on static fund factor exposures.

    Factor_Return_f,t = alpha + beta * Static_Exp_f,t-1 + eps

    Parameters
    ----------
    factor_returns : DataFrame
        FF factor returns (date-indexed).
    static_exposures : DataFrame
        Static factor exposures (date-indexed).

    Returns
    -------
    dict mapping factor -> regression results.
    """
    results = {}
    factors = [c for c in factor_returns.columns if c != "RF"]

    for factor in factors:
        exp_col = f"static_exp_{factor}"
        if exp_col not in static_exposures.columns:
            # Try case variations
            for col in static_exposures.columns:
                if factor.lower() in col.lower():
                    exp_col = col
                    break
            else:
                continue

        df = pd.merge(
            factor_returns[[factor]],
            static_exposures[[exp_col]].shift(1),  # Lagged exposure
            left_index=True, right_index=True, how="inner"
        ).dropna()

        if len(df) < 30:
            continue

        result = ols_nw(df[factor].values, df[[exp_col]].values,
                         max_lag=nw_lags, add_constant=True)

        results[factor] = {
            "beta": result.params[1],
            "se": result.bse[1],
            "t_stat": result.tvalues[1],
            "p_value": result.pvalues[1],
            "r2": result.rsquared,
            "n_obs": int(result.nobs),
        }

    return results


def factor_var_analysis(factor_returns, static_exposures, factors_to_include=None):
    """
    VAR analysis of factor returns and static exposures.

    Parameters
    ----------
    factor_returns : DataFrame
    static_exposures : DataFrame
    factors_to_include : list or None
        Factors to include in VAR.

    Returns
    -------
    dict with VAR results and IRFs.
    """
    if factors_to_include is None:
        factors_to_include = ["Mkt-RF", "SMB", "HML"]

    # Build VAR data
    var_data = factor_returns[factors_to_include].copy()

    # Add static exposures for included factors
    for factor in factors_to_include:
        exp_col = f"static_exp_{factor}"
        if exp_col in static_exposures.columns:
            var_data[exp_col] = static_exposures[exp_col]

    var_data = var_data.dropna()

    if len(var_data) < 50:
        logger.warning("Insufficient data for factor VAR")
        return {}

    var_names = list(var_data.columns)

    try:
        var_est = estimate_var(var_data, var_names, max_lags=6)
        results = {"var": var_est}

        # IRFs: exposure shock -> factor return
        for factor in factors_to_include:
            exp_col = f"static_exp_{factor}"
            if exp_col in var_names:
                irf = impulse_responses(var_est["results"], exp_col, factor)
                results[f"irf_{factor}"] = irf

        return results

    except Exception as e:
        logger.warning(f"Factor VAR failed: {e}")
        return {}


def run_e4(stock_panel=None, ff_factors=None, so_it=None, save_outputs=True):
    """
    Run the full E4 experiment.

    Parameters
    ----------
    stock_panel : DataFrame
        Stock-month panel with factor loadings.
    ff_factors : DataFrame
        FF factor returns.
    so_it : DataFrame
        Static ownership from B1.
    save_outputs : bool
        Save results.

    Returns
    -------
    dict with all E4 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT E4: Factor-Level Bubbles")
    logger.info("=" * 60)

    results = {}

    # --- Load SO_it from B1 if not provided ---
    if so_it is None:
        cached = load_cache("B1_SO_it")
        if cached is not None:
            so_it = cached
        else:
            logger.error("SO_it not available. Run B1 first.")
            return results

    if ff_factors is None:
        logger.error("FF factors not available.")
        return results

    # --- Factor loadings ---
    if stock_panel is not None and "beta" in stock_panel.columns:
        logger.info("Step 1: Computing static factor exposures...")

        # Use available factor loadings
        factor_cols = [c for c in ["beta", "smb_loading", "hml_loading"]
                       if c in stock_panel.columns]

        if factor_cols:
            exposures = compute_static_factor_exposure(
                so_it, stock_panel[["permno", "date"] + factor_cols], factor_cols
            )
            results["exposures"] = exposures
        else:
            # Create simple exposure using market beta only
            if "beta" in stock_panel.columns:
                exp = compute_static_factor_exposure(
                    so_it, stock_panel[["permno", "date", "beta"]], ["beta"]
                )
                exp = exp.rename(columns={"static_exp_beta": "static_exp_Mkt-RF"})
                results["exposures"] = exp

    # --- Factor pressure regressions ---
    if "exposures" in results:
        logger.info("Step 2: Factor pressure regressions...")
        ff = ff_factors.copy()
        if "date" in ff.columns:
            ff = ff.set_index("date")

        pressure = factor_pressure_regressions(ff, results["exposures"])
        results["factor_pressure"] = pressure

        for factor, res in pressure.items():
            logger.info(f"  {factor}: beta = {res['beta']:.4f} (t = {res['t_stat']:.3f})")

    # --- Save ---
    if save_outputs:
        if "factor_pressure" in results:
            rows = []
            for factor, res in results["factor_pressure"].items():
                rows.append({"Factor": factor, **res})
            to_csv(pd.DataFrame(rows), "e4_factor_bubbles.csv")

    logger.info("=" * 60)
    logger.info("E4 COMPLETE")
    logger.info("=" * 60)

    return results
