"""
Experiment C2: Inflow-Return Causality

Tests causal channel from static fund inflows to market returns via
structural VAR and local projections.

Key steps:
1. Structural VAR: [Sharpe, VIX, Delta_F_static, r_t], Cholesky ordering
2. Lag selection via AIC/BIC
3. Impulse response functions: flow shock -> price over 24 months
4. Impact elasticity, half-life, long-run effect
5. Local projections (Jorda) as robustness
6. Regime-dependent IRFs: high-Sharpe vs low-Sharpe

Key outputs: IRF estimates, half-life, regime-dependent dynamics
"""
import logging

import numpy as np
import pandas as pd

from jappelli_experiments.config import NW_LAGS_MONTHLY, IRF_HORIZON
from jappelli_experiments.shared.var_models import (
    estimate_var, impulse_responses, forecast_error_variance_decomposition,
    granger_causality_tests,
)
from jappelli_experiments.shared.local_projections import local_projection_irf, lp_regime_dependent
from jappelli_experiments.shared.statistical_tests import adf_test
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig

logger = logging.getLogger(__name__)
setup_plots()


def prepare_var_data(panel, var_names):
    """
    Prepare data for VAR estimation: check stationarity, transform if needed.

    Parameters
    ----------
    panel : DataFrame
        Aggregate panel.
    var_names : list of str
        Variables to include in VAR.

    Returns
    -------
    DataFrame of stationary variables, dict of transformations applied.
    """
    df = panel[var_names].dropna()
    transforms = {}

    for col in var_names:
        adf = adf_test(df[col])
        if not adf["is_stationary"]:
            logger.info(f"  {col}: non-stationary (ADF p = {adf['p_value']:.3f}), "
                        f"first-differencing")
            df[col] = df[col].diff()
            transforms[col] = "first_difference"
        else:
            transforms[col] = "level"
            logger.info(f"  {col}: stationary (ADF p = {adf['p_value']:.3f})")

    return df.dropna(), transforms


def compute_irf_statistics(irf_df):
    """
    Extract key statistics from impulse response function.

    Parameters
    ----------
    irf_df : DataFrame
        With columns: irf, lower, upper.

    Returns
    -------
    dict with impact, peak, half_life, long_run, cumulative.
    """
    irf = irf_df["irf"].values

    impact = irf[0] if len(irf) > 0 else np.nan
    peak = np.max(np.abs(irf))
    peak_period = np.argmax(np.abs(irf))

    # Half-life: first period where |IRF| < 0.5 * |peak|
    half_life = None
    for h in range(int(peak_period), len(irf)):
        if abs(irf[h]) < 0.5 * peak:
            half_life = h - peak_period
            break
    if half_life is None:
        half_life = len(irf)  # Never reaches half

    # Long-run effect (last period)
    long_run = irf[-1] if len(irf) > 0 else np.nan

    # Cumulative effect
    cumulative = np.sum(irf)

    return {
        "impact": impact,
        "peak": peak,
        "peak_period": peak_period,
        "half_life": half_life,
        "long_run": long_run,
        "cumulative": cumulative,
    }


def run_c2(aggregate_panel, save_outputs=True):
    """
    Run the full C2 experiment.

    Parameters
    ----------
    aggregate_panel : DataFrame
        Aggregate panel with returns, flows, Sharpe, VIX.
    save_outputs : bool
        Save results to disk.

    Returns
    -------
    dict with all C2 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT C2: Inflow-Return Causality")
    logger.info("=" * 60)

    results = {}
    mapper = ConnectionMapper()

    panel = aggregate_panel.copy()

    # Determine available VAR variables
    var_candidates = ["sharpe_t", "vixcls", "static_flow_norm", "vwretd"]
    var_names = [v for v in var_candidates if v in panel.columns]

    if len(var_names) < 3:
        logger.warning(f"Only {len(var_names)} VAR variables available: {var_names}")
        if "vwretd" not in var_names:
            logger.error("Cannot run VAR without return variable.")
            return results

    # --- Step 1: Stationarity and data prep ---
    logger.info("Step 1: Preparing VAR data...")
    var_data, transforms = prepare_var_data(panel, var_names)
    results["transforms"] = transforms
    logger.info(f"  VAR sample: {len(var_data)} observations")

    # --- Step 2: VAR estimation ---
    logger.info("Step 2: Estimating VAR...")
    try:
        var_est = estimate_var(var_data, var_names)
        results["var"] = var_est
        logger.info(f"  Selected lags: {var_est['selected_lags']}")
    except Exception as e:
        logger.error(f"  VAR estimation failed: {e}")
        return results

    # --- Step 3: Impulse response functions ---
    logger.info("Step 3: Computing IRFs...")
    flow_col = "static_flow_norm" if "static_flow_norm" in var_names else var_names[0]
    ret_col = "vwretd" if "vwretd" in var_names else var_names[-1]

    irf = impulse_responses(var_est["results"], flow_col, ret_col,
                             periods=IRF_HORIZON)
    results["irf_flow_to_ret"] = irf
    irf_stats = compute_irf_statistics(irf)
    results["irf_statistics"] = irf_stats
    logger.info(f"  Impact = {irf_stats['impact']:.4f}")
    logger.info(f"  Peak = {irf_stats['peak']:.4f} at period {irf_stats['peak_period']}")
    logger.info(f"  Half-life = {irf_stats['half_life']} periods")

    # --- Step 4: FEVD ---
    logger.info("Step 4: Forecast error variance decomposition...")
    fevd = forecast_error_variance_decomposition(var_est["results"])
    results["fevd"] = fevd
    if ret_col in fevd:
        logger.info(f"  Flow contribution to return variance at h=12: "
                    f"{fevd[ret_col].iloc[min(12, len(fevd[ret_col])-1)].get(flow_col, 'N/A')}")

    # --- Step 5: Granger causality ---
    logger.info("Step 5: Granger causality tests...")
    gc = granger_causality_tests(var_est["results"])
    results["granger"] = gc
    logger.info(f"\n{gc.to_string()}")

    # --- Step 6: Local projections (robustness) ---
    logger.info("Step 6: Local projection IRFs...")
    lp_controls = [c for c in var_names if c not in [flow_col, ret_col]]

    lp_irf = local_projection_irf(
        var_data, ret_col, flow_col, control_cols=lp_controls,
        horizons=IRF_HORIZON, nw_lags=NW_LAGS_MONTHLY
    )
    results["lp_irf"] = lp_irf
    logger.info(f"  LP impact = {lp_irf.loc[0, 'beta']:.4f}")

    # --- Step 7: Regime-dependent IRFs ---
    if "sharpe_t" in panel.columns:
        logger.info("Step 7: Regime-dependent IRFs...")
        var_data_regime = var_data.copy()
        sharpe_aligned = panel["sharpe_t"].reindex(var_data.index)
        var_data_regime["high_sharpe"] = (sharpe_aligned > sharpe_aligned.median()).astype(float)
        var_data_regime = var_data_regime.dropna()

        lp_regime = lp_regime_dependent(
            var_data_regime, ret_col, flow_col, "high_sharpe",
            control_cols=lp_controls, horizons=IRF_HORIZON
        )
        results["lp_regime"] = lp_regime

    # --- Save ---
    if save_outputs:
        to_csv(irf, "c2_irf_flow_to_return.csv")
        to_csv(pd.DataFrame([irf_stats]), "c2_irf_statistics.csv")
        to_csv(gc, "c2_granger_causality.csv")
        to_csv(lp_irf, "c2_local_projection_irf.csv")

        # Figure: IRFs
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # VAR IRF
        ax = axes[0]
        ax.plot(irf.index, irf["irf"], color="#2C3E50", linewidth=2)
        if not irf["lower"].isna().all():
            ax.fill_between(irf.index, irf["lower"], irf["upper"],
                            alpha=0.2, color="#2C3E50")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Months")
        ax.set_ylabel("Response")
        ax.set_title("VAR IRF: Flow Shock → Market Return")

        # LP IRF
        ax = axes[1]
        ax.plot(lp_irf["horizon"], lp_irf["beta"], color="#E74C3C", linewidth=2)
        ax.fill_between(lp_irf["horizon"], lp_irf["ci_lower"], lp_irf["ci_upper"],
                        alpha=0.2, color="#E74C3C")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Months")
        ax.set_ylabel("Response")
        ax.set_title("Local Projection IRF: Flow Shock → Market Return")

        fig.tight_layout()
        save_fig(fig, "c2_irfs.pdf")

    logger.info("=" * 60)
    logger.info("C2 COMPLETE")
    logger.info("=" * 60)

    return results
