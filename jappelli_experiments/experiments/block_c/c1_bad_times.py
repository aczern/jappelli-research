"""
Experiment C1: Bad-Times Amplification

Tests that the bubble component becomes more sensitive to static investors
during bad times (low Sharpe ratios), because dynamic investors retreat.

Key steps:
1. Rolling-window regression: V_t on theta_t, save time-varying beta_t
2. Regress beta_t on lagged Sharpe_t (amplification coefficient)
3. Kalman filter state-space model: beta follows random walk
4. Sharpe quartile bins: separate regressions, test equality

Key outputs: Time-varying beta, amplification coefficient, regime estimates
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from jappelli_experiments.config import (
    NW_LAGS_MONTHLY, NW_LAGS_QUARTERLY, ROLLING_WINDOW_MONTHS,
)
from jappelli_experiments.shared.rolling_estimation import rolling_ols, rolling_sharpe
from jappelli_experiments.shared.kalman import tvp_regression
from jappelli_experiments.shared.newey_west import ols_nw, nw_t_test
from jappelli_experiments.shared.statistical_tests import chow_test
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cache import save_cache, load_cache

logger = logging.getLogger(__name__)
setup_plots()


def rolling_bubble_sensitivity(V_t, theta_t, window=ROLLING_WINDOW_MONTHS):
    """
    Estimate time-varying sensitivity of V_t to theta_t via rolling OLS.

    V_t = alpha_t + beta_t * theta_t + eps_t

    Parameters
    ----------
    V_t : Series
        Bubble component (from A1).
    theta_t : Series
        Aggregate static fund allocation.
    window : int
        Rolling window size.

    Returns
    -------
    DataFrame with time-varying beta_t and R-squared.
    """
    df = pd.DataFrame({"V": V_t, "theta": theta_t}).dropna()
    result = rolling_ols(df, "V", ["theta"], window=window)
    return result


def amplification_test(beta_t, sharpe_t, nw_lags=NW_LAGS_QUARTERLY):
    """
    Test whether sensitivity increases during bad times.

    beta_t = gamma_0 + gamma_1 * Sharpe_{t-1} + eps

    If gamma_1 < 0: amplification in bad times (low Sharpe -> high beta).

    Parameters
    ----------
    beta_t : Series
        Time-varying beta from rolling regression.
    sharpe_t : Series
        Sharpe ratio series.
    nw_lags : int
        Newey-West lag truncation.

    Returns
    -------
    dict with amplification coefficient and diagnostics.
    """
    df = pd.DataFrame({"beta": beta_t, "sharpe": sharpe_t}).dropna()
    df["sharpe_lag"] = df["sharpe"].shift(1)
    df = df.dropna()

    result = ols_nw(df["beta"].values, df[["sharpe_lag"]].values,
                     max_lag=nw_lags, add_constant=True)

    return {
        "gamma_0": result.params[0],
        "gamma_1": result.params[1],
        "se_gamma1": result.bse[1],
        "t_stat": result.tvalues[1],
        "p_value": result.pvalues[1],
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
        "interpretation": (
            "Amplification in bad times"
            if result.params[1] < 0 and result.pvalues[1] < 0.10
            else "No significant amplification"
        ),
    }


def kalman_tvp_bubble_sensitivity(V_t, theta_t):
    """
    Kalman filter state-space model for time-varying beta.

    State: beta_t = beta_{t-1} + eta_t
    Observation: V_t = beta_t * theta_t + eps_t

    Parameters
    ----------
    V_t : Series
        Bubble component.
    theta_t : Series
        Theta series.

    Returns
    -------
    dict with filtered and smoothed beta series.
    """
    df = pd.DataFrame({"V": V_t, "theta": theta_t}).dropna()

    y = df["V"].values
    X = df[["theta"]].values

    result = tvp_regression(y, sm.add_constant(X), col_names=["const", "beta_theta"])
    return result


def sharpe_regime_regressions(V_t, theta_t, sharpe_t, n_regimes=4):
    """
    Estimate V_t ~ theta_t separately within Sharpe ratio regimes.

    Parameters
    ----------
    n_regimes : int
        Number of Sharpe quartiles.

    Returns
    -------
    DataFrame with beta per regime and equality test.
    """
    df = pd.DataFrame({"V": V_t, "theta": theta_t, "sharpe": sharpe_t}).dropna()
    df["regime"] = pd.qcut(df["sharpe"], n_regimes, labels=False) + 1

    results = []
    for regime in range(1, n_regimes + 1):
        sub = df[df["regime"] == regime]
        if len(sub) < 20:
            continue
        res = ols_nw(sub["V"].values, sub[["theta"]].values,
                      max_lag=NW_LAGS_QUARTERLY, add_constant=True)
        results.append({
            "regime": regime,
            "sharpe_range": f"Q{regime}",
            "mean_sharpe": sub["sharpe"].mean(),
            "beta": res.params[1],
            "se": res.bse[1],
            "t_stat": res.tvalues[1],
            "r2": res.rsquared,
            "n_obs": len(sub),
        })

    return pd.DataFrame(results)


def run_c1(aggregate_panel, V_t=None, save_outputs=True):
    """
    Run the full C1 experiment.

    Parameters
    ----------
    aggregate_panel : DataFrame
        Aggregate monthly panel with theta_t and Sharpe ratio.
    V_t : Series or None
        Bubble component from A1. If None, loads from cache.
    save_outputs : bool
        Save results to disk.

    Returns
    -------
    dict with all C1 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT C1: Bad-Times Amplification")
    logger.info("=" * 60)

    results = {}
    mapper = ConnectionMapper()

    # --- Load V_t from A1 ---
    if V_t is None:
        cached_vt = load_cache("A1_V_t")
        if cached_vt is not None:
            V_t = cached_vt["V_t"]
        else:
            raise ValueError("V_t not provided and not found in cache. Run A1 first.")

    # Validate consistency
    mapper.validate_input("C1", "V_t", V_t)

    panel = aggregate_panel.copy()

    if "theta_t" not in panel.columns:
        raise ValueError("theta_t not in aggregate panel")

    # --- Step 1: Rolling bubble sensitivity ---
    logger.info("Step 1: Rolling-window V_t ~ theta_t regression...")
    rolling_result = rolling_bubble_sensitivity(V_t, panel["theta_t"])
    results["rolling_beta"] = rolling_result
    beta_t = rolling_result["theta"].dropna()
    logger.info(f"  Mean beta = {beta_t.mean():.4f}, SD = {beta_t.std():.4f}")

    # --- Step 2: Amplification test ---
    if "sharpe_t" in panel.columns:
        logger.info("Step 2: Amplification test (beta_t ~ Sharpe_{t-1})...")
        amp = amplification_test(beta_t, panel["sharpe_t"])
        results["amplification"] = amp
        logger.info(f"  gamma_1 = {amp['gamma_1']:.4f} (t = {amp['t_stat']:.3f})")
        logger.info(f"  {amp['interpretation']}")

    # --- Step 3: Kalman filter TVP ---
    logger.info("Step 3: Kalman filter TVP model...")
    try:
        tvp = kalman_tvp_bubble_sensitivity(V_t, panel["theta_t"])
        results["kalman_tvp"] = tvp
        logger.info(f"  Log-likelihood = {tvp['loglik']:.2f}")
    except Exception as e:
        logger.warning(f"  Kalman filter failed: {e}")

    # --- Step 4: Regime regressions ---
    if "sharpe_t" in panel.columns:
        logger.info("Step 4: Sharpe quartile regime regressions...")
        regime_results = sharpe_regime_regressions(V_t, panel["theta_t"], panel["sharpe_t"])
        results["regime_regressions"] = regime_results
        logger.info(f"\n{regime_results.to_string()}")

    # Register outputs
    mapper.register_output("C1", "time_varying_beta", beta_t)
    if "amplification" in results:
        mapper.register_output("C1", "amplification_coef",
                               pd.Series(results["amplification"]["gamma_1"],
                                         name="gamma_1"))

    # --- Save ---
    if save_outputs:
        if "amplification" in results:
            to_csv(pd.DataFrame([results["amplification"]]), "c1_amplification.csv")
        if "regime_regressions" in results:
            to_csv(results["regime_regressions"], "c1_regime_regressions.csv")

        # Figure: time-varying beta
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax = axes[0]
        ax.plot(beta_t.index, beta_t.values, color="#2C3E50", linewidth=1.5)
        ax.axhline(y=beta_t.mean(), color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Time-Varying Sensitivity of V_t to theta_t")
        ax.set_ylabel("beta_t")

        if "sharpe_t" in panel.columns:
            ax = axes[1]
            ax.scatter(panel["sharpe_t"].values, beta_t.reindex(panel.index).values,
                       alpha=0.3, s=10, color="#E74C3C")
            ax.set_xlabel("Sharpe Ratio")
            ax.set_ylabel("beta_t")
            ax.set_title("Amplification: beta_t vs Sharpe Ratio")

        fig.tight_layout()
        save_fig(fig, "c1_bad_times.pdf")

    logger.info("=" * 60)
    logger.info("C1 COMPLETE")
    logger.info("=" * 60)

    return results
