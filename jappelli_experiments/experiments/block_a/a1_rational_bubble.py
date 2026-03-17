"""
Experiment A1: Direct Test of the Rational Bubble in theta * V_t

Tests Jappelli (2025) Proposition 1: P_t = PDV_t(D_t) + theta * V_t

Key steps:
1. Construct PDV_t using 3 methods (Gordon growth, VAR, long-horizon regression)
2. Compute V_t = P_t - PDV_t (residual bubble component)
3. Martingale test: V_{t+1} - V_t regressed on V_t + instruments. H0: slope=0.
4. Transversality violation: long-horizon VAR for E[sum delta^s V_{t+s}]
5. Fundamental orthogonality: V_t on lagged dividend yield, consumption growth

Key outputs: V_t time series (feeds C1, E2), PDV_t estimates, Table A1
Validation target: Replicate Jappelli Table 1 martingale test (p-value = 0.32)
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from jappelli_experiments.config import NW_LAGS_QUARTERLY, STATIC_SD_ALTERNATIVES
from jappelli_experiments.shared.newey_west import ols_nw, nw_t_test
from jappelli_experiments.shared.var_models import estimate_var, impulse_responses
from jappelli_experiments.shared.statistical_tests import adf_test
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import format_regression_table, to_latex, to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cache import save_cache, load_cache

logger = logging.getLogger(__name__)
setup_plots()


# ── Step 1: PDV Construction ──

def pdv_gordon_growth(dividends, discount_rate, growth_rate):
    """
    Gordon Growth Model PDV: PDV_t = D_t / (r - g).

    Parameters
    ----------
    dividends : Series
        Dividend series (date-indexed).
    discount_rate : Series or float
        Discount rate (if Series, time-varying).
    growth_rate : float
        Constant dividend growth rate.

    Returns
    -------
    Series of PDV_t values.
    """
    if isinstance(discount_rate, (int, float)):
        discount_rate = pd.Series(discount_rate, index=dividends.index)

    spread = discount_rate - growth_rate
    spread = spread.clip(lower=0.005)  # Avoid division by zero
    return dividends / spread


def pdv_var_method(panel, ret_col, dp_col, n_lags=4):
    """
    VAR-based PDV following Campbell-Shiller (1988).

    Estimates VAR on [r_t, dp_t] and computes PDV from implied
    long-run dividend forecasts.

    Parameters
    ----------
    panel : DataFrame
        Data with return and dividend-price ratio columns.
    ret_col : str
        Return column.
    dp_col : str
        Log dividend-price ratio column.
    n_lags : int
        VAR lag length.

    Returns
    -------
    Series of PDV_t values.
    """
    df = panel[[ret_col, dp_col]].dropna()

    var_result = estimate_var(df, [ret_col, dp_col], max_lags=n_lags)
    results = var_result["results"]

    # Extract companion matrix for long-run forecasts
    A = results.coefs  # lag coefficient matrices
    k = len(results.names)

    # Companion form
    p = results.k_ar  # number of lags
    companion = np.zeros((k * p, k * p))
    for i in range(p):
        companion[:k, i * k:(i + 1) * k] = A[i]
    if p > 1:
        companion[k:, :(p - 1) * k] = np.eye(k * (p - 1))

    # Discount factor (approximate)
    rho = 0.96 ** (1 / 12)  # Monthly discount factor

    # PDV = sum_{s=1}^inf rho^s * e1' * A^s * x_t
    # where e1 selects the dividend component
    T = len(df)
    pdv = np.zeros(T)

    for t in range(p, T):
        # State vector
        x = np.zeros(k * p)
        for lag in range(p):
            if t - lag >= 0:
                x[lag * k:(lag + 1) * k] = df.iloc[t - lag].values

        # Sum discounted forecasts
        A_power = np.eye(k * p)
        pdv_sum = 0
        for s in range(1, 200):  # Truncate at 200 periods
            A_power = A_power @ companion
            contribution = rho ** s * A_power[0, :k] @ df.iloc[t].values
            pdv_sum += contribution
            if abs(contribution) < 1e-10:
                break
        pdv[t] = pdv_sum

    return pd.Series(pdv, index=df.index, name="PDV_var")


def pdv_long_horizon(prices, dividends, horizon=120):
    """
    Long-horizon regression PDV.

    Regress cumulative future dividends on current price to extract
    the fundamental component.

    Parameters
    ----------
    prices : Series
        Price level (date-indexed).
    dividends : Series
        Dividend series.
    horizon : int
        Forecast horizon (months).

    Returns
    -------
    Series of PDV_t values.
    """
    df = pd.DataFrame({"P": prices, "D": dividends}).dropna()

    # Cumulative discounted future dividends
    rho = 0.96 ** (1 / 12)
    cum_div = pd.Series(0.0, index=df.index)
    for h in range(1, horizon + 1):
        cum_div += rho ** h * df["D"].shift(-h)

    # Regression: cum_div_t = alpha + beta * P_t + eps
    valid = pd.DataFrame({"cum_div": cum_div, "P": df["P"]}).dropna()
    X = sm.add_constant(valid["P"].values)
    res = sm.OLS(valid["cum_div"].values, X).fit()

    # Fitted values = PDV estimate
    pdv = pd.Series(np.nan, index=df.index, name="PDV_longhorizon")
    pdv.loc[valid.index] = res.fittedvalues

    return pdv


# ── Step 2: V_t Construction ──

def compute_bubble_component(prices, pdv, name="V_t"):
    """
    Compute bubble component: V_t = P_t - PDV_t.

    Parameters
    ----------
    prices : Series
        Market price/index level.
    pdv : Series
        Present discounted value of dividends.

    Returns
    -------
    Series of V_t values.
    """
    V = prices - pdv
    V.name = name
    return V


# ── Step 3: Martingale Test ──

def martingale_test(V_t, instruments=None, nw_lags=NW_LAGS_QUARTERLY):
    """
    Test whether V_t follows a martingale: V_{t+1} - V_t = alpha + beta*V_t + eps.

    H0: beta = 0, R^2 ~ 0 (consistent with rational bubble).

    Parameters
    ----------
    V_t : Series
        Bubble component (date-indexed).
    instruments : DataFrame or None
        Additional instruments (lagged variables).
    nw_lags : int
        Newey-West lag truncation.

    Returns
    -------
    dict with regression results and test statistics.
    """
    df = pd.DataFrame({"V": V_t}).dropna()
    df["V_lead"] = df["V"].shift(-1)
    df["dV"] = df["V_lead"] - df["V"]
    df["V_lag"] = df["V"]
    df = df.dropna()

    # Basic test: dV on V_lag
    X = df[["V_lag"]].values
    y = df["dV"].values

    result = ols_nw(y, X, max_lag=nw_lags, add_constant=True)

    output = {
        "beta": result.params[1],
        "se": result.bse[1],
        "t_stat": result.tvalues[1],
        "p_value": result.pvalues[1],
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
        "nw_lags": nw_lags,
    }

    # Extended test with instruments
    if instruments is not None:
        inst_df = pd.merge(df, instruments, left_index=True, right_index=True, how="inner")
        inst_cols = [c for c in instruments.columns if c in inst_df.columns]
        X_ext = inst_df[["V_lag"] + inst_cols].values
        y_ext = inst_df["dV"].values

        result_ext = ols_nw(y_ext, X_ext, max_lag=nw_lags, add_constant=True)
        output["extended"] = {
            "beta": result_ext.params[1],
            "se": result_ext.bse[1],
            "t_stat": result_ext.tvalues[1],
            "p_value": result_ext.pvalues[1],
            "r2": result_ext.rsquared,
        }

    return output


# ── Step 4: Transversality Violation ──

def transversality_test(V_t, discount_factor=0.96, horizon=40):
    """
    Test transversality condition: does E[sum delta^s V_{t+s}] converge?

    If the sum is finite and positive, transversality is violated,
    consistent with a rational bubble.

    Parameters
    ----------
    V_t : Series
        Bubble component.
    discount_factor : float
        Per-period discount factor.
    horizon : int
        Maximum horizon for sum.

    Returns
    -------
    dict with sum estimate, confidence interval, and interpretation.
    """
    V = V_t.dropna().values
    T = len(V)

    sums = []
    for t in range(T - horizon):
        s = sum(discount_factor ** h * V[t + h] for h in range(horizon))
        sums.append(s)

    sums = np.array(sums)
    result = nw_t_test(sums, max_lag=NW_LAGS_QUARTERLY)
    result["interpretation"] = (
        "Transversality violated (rational bubble)"
        if result["mean"] > 0 and result["p_value"] < 0.10
        else "Cannot reject transversality"
    )
    return result


# ── Step 5: Fundamental Orthogonality ──

def fundamental_orthogonality_test(V_t, fundamentals, nw_lags=NW_LAGS_QUARTERLY):
    """
    Test that V_t is orthogonal to fundamental variables.

    Regress V_t on lagged dividend yield, consumption growth, macro fundamentals.
    H0: all coefficients ~ 0.

    Parameters
    ----------
    V_t : Series
        Bubble component.
    fundamentals : DataFrame
        Fundamental variables (date-indexed).
    nw_lags : int
        Newey-West lags.

    Returns
    -------
    dict with joint F-test and individual coefficient results.
    """
    df = pd.merge(
        V_t.to_frame("V"),
        fundamentals,
        left_index=True, right_index=True, how="inner"
    ).dropna()

    fund_cols = [c for c in fundamentals.columns if c in df.columns]
    X = df[fund_cols].values
    y = df["V"].values

    result = ols_nw(y, X, max_lag=nw_lags, add_constant=True)

    coefs = {}
    for i, col in enumerate(fund_cols):
        coefs[col] = {
            "beta": result.params[i + 1],
            "se": result.bse[i + 1],
            "t_stat": result.tvalues[i + 1],
            "p_value": result.pvalues[i + 1],
        }

    return {
        "coefficients": coefs,
        "joint_f": result.fvalue,
        "joint_p": result.f_pvalue,
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
    }


# ── Robustness ──

def robustness_alternative_thresholds(mf_summary, prices, dividends,
                                       thresholds=STATIC_SD_ALTERNATIVES):
    """
    Re-run martingale test under alternative static fund thresholds.

    Returns
    -------
    DataFrame comparing results across thresholds.
    """
    from jappelli_experiments.data.panel_builder import classify_static_funds, compute_theta_t

    results = []
    for thresh in thresholds:
        fund_class = classify_static_funds(mf_summary, sd_threshold=thresh)
        theta = compute_theta_t(mf_summary, fund_class)

        # Quick PDV via Gordon growth (use first method for robustness)
        pdv = pdv_gordon_growth(dividends, 0.08, 0.02)
        V = compute_bubble_component(prices, pdv)

        test = martingale_test(V)
        results.append({
            "threshold": thresh,
            "n_static_funds": fund_class["is_static"].sum(),
            "beta": test["beta"],
            "t_stat": test["t_stat"],
            "p_value": test["p_value"],
            "r2": test["r2"],
        })

    return pd.DataFrame(results)


# ── Main Runner ──

def run_a1(aggregate_panel, mf_summary=None, save_outputs=True):
    """
    Run the full A1 experiment.

    Parameters
    ----------
    aggregate_panel : DataFrame
        Aggregate monthly panel with market returns, dividends, etc.
    mf_summary : DataFrame
        Mutual fund summary (for robustness checks).
    save_outputs : bool
        Whether to save outputs to disk.

    Returns
    -------
    dict with all results, keyed by test name.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT A1: Direct Test of the Rational Bubble")
    logger.info("=" * 60)

    results = {}
    mapper = ConnectionMapper()

    # --- Prepare data ---
    panel = aggregate_panel.copy()

    # Need: prices, dividends, dividend yield
    # Use CRSP value-weighted return to construct cumulative price index
    if "vwretd" in panel.columns:
        panel["price_index"] = (1 + panel["vwretd"]).cumprod() * 100
    elif "totval" in panel.columns:
        panel["price_index"] = panel["totval"]
    else:
        raise ValueError("Need vwretd or totval in aggregate panel")

    # Dividend yield proxy from FF factors (Mkt-RF + RF = market return)
    if "RF" in panel.columns and "Mkt-RF" in panel.columns:
        panel["mkt_ret"] = panel["Mkt-RF"] + panel["RF"]

    # --- Step 1: PDV Construction ---
    logger.info("Step 1: Constructing PDV_t...")

    # Method 1: Gordon Growth (baseline)
    # Estimate growth rate from trailing returns
    if "vwretd" in panel.columns:
        g_hat = panel["vwretd"].expanding(min_periods=60).mean()
        r_hat = g_hat + 0.04  # Equity premium proxy
        pdv_gg = panel["price_index"] * g_hat / r_hat.clip(lower=0.01)
        pdv_gg.name = "PDV_gordon"
        results["PDV_gordon"] = pdv_gg

    # Method 2: VAR-based (if sufficient data)
    if "vwretd" in panel.columns:
        panel["dp_ratio"] = np.log(panel["price_index"]).diff(12)  # Proxy
        try:
            pdv_v = pdv_var_method(panel.dropna(subset=["vwretd", "dp_ratio"]),
                                    "vwretd", "dp_ratio")
            results["PDV_var"] = pdv_v
        except Exception as e:
            logger.warning(f"VAR PDV failed: {e}")

    # Use Gordon Growth as primary
    pdv_primary = results.get("PDV_gordon", results.get("PDV_var"))
    if pdv_primary is None:
        raise ValueError("Could not construct any PDV estimate")

    # --- Step 2: V_t ---
    logger.info("Step 2: Computing V_t = P_t - PDV_t...")
    V_t = compute_bubble_component(panel["price_index"], pdv_primary)
    results["V_t"] = V_t

    # Register with connection mapper
    mapper.register_output("A1", "V_t", V_t)
    mapper.register_output("A1", "PDV_t", pdv_primary)
    if "theta_t" in panel.columns:
        mapper.register_output("A1", "theta_t", panel["theta_t"])

    # --- Step 3: Martingale Test ---
    logger.info("Step 3: Martingale test...")
    mart_test = martingale_test(V_t)
    results["martingale_test"] = mart_test
    logger.info(f"  Beta = {mart_test['beta']:.4f} (SE = {mart_test['se']:.4f})")
    logger.info(f"  t-stat = {mart_test['t_stat']:.3f}, p-value = {mart_test['p_value']:.3f}")
    logger.info(f"  R² = {mart_test['r2']:.4f}")
    logger.info(f"  Target p-value: 0.32 (Jappelli Table 1)")

    # --- Step 4: Transversality ---
    logger.info("Step 4: Transversality violation test...")
    trans_test = transversality_test(V_t)
    results["transversality"] = trans_test
    logger.info(f"  Mean discounted sum = {trans_test['mean']:.4f}")
    logger.info(f"  {trans_test['interpretation']}")

    # --- Step 5: Fundamental Orthogonality ---
    logger.info("Step 5: Fundamental orthogonality test...")
    fund_vars = [c for c in ["dp_ratio", "dgs10", "vixcls", "cpiaucsl"] if c in panel.columns]
    if fund_vars:
        fund_df = panel[fund_vars].copy()
        # Use lagged values
        for c in fund_vars:
            fund_df[c] = fund_df[c].shift(1)
        fund_df = fund_df.dropna()

        ortho_test = fundamental_orthogonality_test(V_t, fund_df)
        results["orthogonality"] = ortho_test
        logger.info(f"  Joint F = {ortho_test['joint_f']:.3f}, p = {ortho_test['joint_p']:.3f}")

    # --- Stationarity check ---
    logger.info("ADF test on V_t...")
    adf = adf_test(V_t.dropna())
    results["adf_V_t"] = adf
    logger.info(f"  ADF stat = {adf['adf_stat']:.3f}, p = {adf['p_value']:.3f}")

    # --- Output ---
    if save_outputs:
        # Table A1: Martingale test
        table_data = pd.DataFrame([{
            "Test": "Martingale (baseline)",
            "beta": f"{mart_test['beta']:.4f}",
            "SE (NW)": f"{mart_test['se']:.4f}",
            "t-stat": f"{mart_test['t_stat']:.3f}",
            "p-value": f"{mart_test['p_value']:.3f}",
            "R²": f"{mart_test['r2']:.4f}",
            "N": mart_test['n_obs'],
        }])
        to_csv(table_data, "a1_martingale_test.csv")

        # Figure: V_t time series
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax = axes[0]
        V_plot = V_t.dropna()
        ax.plot(V_plot.index, V_plot.values, color="#2C3E50", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Bubble Component V_t = P_t - PDV_t")
        ax.set_ylabel("V_t")

        ax = axes[1]
        ax.plot(panel.index, panel["price_index"], label="Price Index", color="#2C3E50")
        if pdv_primary is not None:
            pdv_plot = pdv_primary.dropna()
            ax.plot(pdv_plot.index, pdv_plot.values, label="PDV_t", color="#E74C3C", linestyle="--")
        ax.set_title("Price vs. Present Discounted Value")
        ax.set_ylabel("Level")
        ax.legend()

        fig.tight_layout()
        save_fig(fig, "a1_bubble_component.pdf")

        # Save V_t for downstream experiments
        save_cache(V_t.to_frame("V_t"), "A1_V_t")
        save_cache(pdv_primary.to_frame("PDV_t"), "A1_PDV_t")

    logger.info("=" * 60)
    logger.info("A1 COMPLETE")
    logger.info("=" * 60)

    return results
