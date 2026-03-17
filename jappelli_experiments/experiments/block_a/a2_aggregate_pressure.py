"""
Experiment A2: Aggregate Price Pressure from Static Fund Inflows

Tests whether static fund inflows create aggregate price pressure:
    Delta_P_t = alpha + beta * (Delta_F_static / MV_t) + controls + eps

Key steps:
1. Compute static fund flows: Delta_F_static / MV_t
2. OLS regression with Newey-West standard errors
3. IV regression: S&P reconstitution, 401(k) timing, regulatory events
4. Event study around index inclusion/exclusion
5. First-stage diagnostics (F > 10, Hansen J)

Key outputs: Beta (flow elasticity), IV estimates, event study CARs
Reference: Translate Jiang main_analysis.do Newey-West regressions
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from jappelli_experiments.config import NW_LAGS_MONTHLY, NW_LAGS_QUARTERLY
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.iv_regression import iv_2sls, weak_instrument_test
from jappelli_experiments.shared.statistical_tests import regression_diagnostics
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cache import save_cache

logger = logging.getLogger(__name__)
setup_plots()


def compute_flow_pressure(aggregate_flows, market_value):
    """
    Compute flow pressure variable: Delta_F_static / MV_t.

    Parameters
    ----------
    aggregate_flows : DataFrame
        With static_flow, dynamic_flow columns (date-indexed).
    market_value : Series
        Total market capitalization (date-indexed).

    Returns
    -------
    DataFrame with normalized flow measures.
    """
    df = pd.DataFrame({
        "static_flow": aggregate_flows["static_flow"],
        "dynamic_flow": aggregate_flows["dynamic_flow"],
        "total_flow": aggregate_flows["total_flow"],
        "mv": market_value,
    }).dropna()

    df["static_flow_norm"] = df["static_flow"] / df["mv"]
    df["dynamic_flow_norm"] = df["dynamic_flow"] / df["mv"]
    df["total_flow_norm"] = df["total_flow"] / df["mv"]

    return df


def ols_flow_elasticity(panel, ret_col="vwretd", flow_col="static_flow_norm",
                         control_cols=None, nw_lags=NW_LAGS_MONTHLY):
    """
    OLS regression of returns on normalized flows with NW standard errors.

    Parameters
    ----------
    panel : DataFrame
        Aggregate panel with returns and flow measures.
    ret_col : str
        Return column.
    flow_col : str
        Flow pressure column.
    control_cols : list of str or None
        Control variables.
    nw_lags : int
        Newey-West lag truncation.

    Returns
    -------
    dict with regression results.
    """
    if control_cols is None:
        control_cols = []

    df = panel[[ret_col, flow_col] + control_cols].dropna()

    X_cols = [flow_col] + control_cols
    result = ols_nw(df[ret_col].values, df[X_cols].values,
                     max_lag=nw_lags, add_constant=True)

    X_full = sm.add_constant(df[X_cols].values)
    diag = regression_diagnostics(result, X_full)

    return {
        "beta": result.params[1],
        "se": result.bse[1],
        "t_stat": result.tvalues[1],
        "p_value": result.pvalues[1],
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
        "full_results": result,
        "diagnostics": diag,
    }


def iv_flow_elasticity(panel, ret_col, endog_col, instrument_cols,
                         control_cols=None):
    """
    IV regression of returns on flows using excluded instruments.

    Parameters
    ----------
    panel : DataFrame
    ret_col : str
        Dependent variable.
    endog_col : str
        Endogenous flow variable.
    instrument_cols : list of str
        Excluded instruments (e.g., S&P reconstitution dates).
    control_cols : list of str or None
        Exogenous controls.

    Returns
    -------
    dict with IV results and diagnostics.
    """
    if control_cols is None:
        control_cols = []

    iv_result = iv_2sls(
        panel, ret_col, [endog_col], control_cols, instrument_cols
    )

    # Weak instrument check
    for col, fs in iv_result["first_stage"].items():
        wi = weak_instrument_test(fs["f_stat"])
        iv_result["diagnostics"]["weak_instrument"] = wi

    return iv_result


def event_study_index_inclusion(returns, events, window=(-10, 10)):
    """
    Event study around S&P 500 index additions/deletions.

    Parameters
    ----------
    returns : DataFrame
        Stock-level daily returns with permno and date.
    events : DataFrame
        Event data with permno, event_date, event_type ('add' or 'drop').
    window : tuple
        (pre_days, post_days) around event.

    Returns
    -------
    DataFrame with cumulative abnormal returns by event type.
    """
    results = []

    for _, event in events.iterrows():
        permno = event.get("permno")
        event_date = event.get("event_date", event.get("date"))
        event_type = event.get("event_type", event.get("type", "add"))

        # Get stock returns around event
        stock_rets = returns[returns["permno"] == permno].copy()
        stock_rets = stock_rets.sort_values("date")

        # Find event date in trading calendar
        dates = stock_rets["date"].values
        event_idx = np.searchsorted(dates, np.datetime64(event_date))

        if event_idx < abs(window[0]) or event_idx + window[1] >= len(dates):
            continue

        # Extract window returns
        start = event_idx + window[0]
        end = event_idx + window[1] + 1
        window_rets = stock_rets.iloc[start:end]["ret"].values

        # Cumulative return
        car = np.cumprod(1 + window_rets) - 1

        results.append({
            "permno": permno,
            "event_date": event_date,
            "event_type": event_type,
            "car": car[-1],
            "car_series": car,
        })

    return pd.DataFrame(results)


def run_a2(aggregate_panel, aggregate_flows=None, market_value=None,
           sp500_events=None, save_outputs=True):
    """
    Run the full A2 experiment.

    Parameters
    ----------
    aggregate_panel : DataFrame
        Aggregate monthly panel.
    aggregate_flows : DataFrame
        Static/dynamic fund flow aggregates.
    market_value : Series
        Total market capitalization.
    sp500_events : DataFrame
        S&P 500 addition/deletion events for IV.
    save_outputs : bool
        Whether to save outputs.

    Returns
    -------
    dict with all results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT A2: Aggregate Price Pressure")
    logger.info("=" * 60)

    results = {}
    mapper = ConnectionMapper()

    # --- Prepare flow data ---
    if aggregate_flows is not None and market_value is not None:
        flow_data = compute_flow_pressure(aggregate_flows, market_value)
        panel = aggregate_panel.join(flow_data, how="inner")
    else:
        panel = aggregate_panel.copy()
        logger.warning("No separate flow data provided; using panel columns")

    # --- OLS: Baseline flow elasticity ---
    logger.info("Step 1: OLS flow elasticity...")

    if "static_flow_norm" in panel.columns:
        # Model 1: Univariate
        ols_1 = ols_flow_elasticity(panel, "vwretd", "static_flow_norm")
        results["ols_univariate"] = ols_1
        logger.info(f"  Univariate beta = {ols_1['beta']:.4f} (t = {ols_1['t_stat']:.3f})")

        # Model 2: With controls
        controls = [c for c in ["dgs10", "vixcls", "fedfunds"] if c in panel.columns]
        if controls:
            ols_2 = ols_flow_elasticity(panel, "vwretd", "static_flow_norm", controls)
            results["ols_controls"] = ols_2
            logger.info(f"  With controls beta = {ols_2['beta']:.4f} (t = {ols_2['t_stat']:.3f})")

        # Model 3: Horse race (static vs dynamic)
        if "dynamic_flow_norm" in panel.columns:
            both_cols = ["static_flow_norm", "dynamic_flow_norm"]
            df_both = panel[["vwretd"] + both_cols].dropna()
            horse = ols_nw(df_both["vwretd"].values, df_both[both_cols].values,
                           max_lag=NW_LAGS_MONTHLY, add_constant=True)
            results["horse_race"] = {
                "beta_static": horse.params[1],
                "se_static": horse.bse[1],
                "beta_dynamic": horse.params[2],
                "se_dynamic": horse.bse[2],
                "r2": horse.rsquared,
            }
            logger.info(f"  Horse race: beta_static={horse.params[1]:.4f}, "
                        f"beta_dynamic={horse.params[2]:.4f}")

    # --- IV regression ---
    if sp500_events is not None and "static_flow_norm" in panel.columns:
        logger.info("Step 2: IV flow elasticity...")

        # Construct instrument: number of S&P additions/deletions per month
        events = sp500_events.copy()
        if "date" in events.columns:
            events["month"] = pd.to_datetime(events["date"]) + pd.offsets.MonthEnd(0)
            iv_counts = events.groupby("month").size().rename("n_reconstitutions")
            panel = panel.join(iv_counts, how="left")
            panel["n_reconstitutions"] = panel["n_reconstitutions"].fillna(0)

            try:
                iv_result = iv_flow_elasticity(
                    panel.dropna(subset=["vwretd", "static_flow_norm", "n_reconstitutions"]),
                    "vwretd", "static_flow_norm", ["n_reconstitutions"]
                )
                results["iv"] = iv_result
                logger.info(f"  IV results: {iv_result['diagnostics']}")
            except Exception as e:
                logger.warning(f"  IV estimation failed: {e}")

    # Register outputs
    if "static_flow_norm" in panel.columns:
        mapper.register_output("A2", "flow_definitions", panel[["static_flow_norm"]].dropna())

    # --- Save outputs ---
    if save_outputs and results:
        rows = []
        for name, r in results.items():
            if isinstance(r, dict) and "beta" in r:
                rows.append({
                    "Model": name,
                    "beta": r.get("beta", ""),
                    "SE": r.get("se", ""),
                    "t-stat": r.get("t_stat", ""),
                    "p-value": r.get("p_value", ""),
                    "R²": r.get("r2", ""),
                    "N": r.get("n_obs", ""),
                })
        if rows:
            to_csv(pd.DataFrame(rows), "a2_flow_elasticity.csv")

    logger.info("=" * 60)
    logger.info("A2 COMPLETE")
    logger.info("=" * 60)

    return results
