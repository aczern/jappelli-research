"""
Experiment E1: International Validation

Replicates A2 aggregate price pressure test for international markets
(UK, Canada, Japan, Eurozone) to assess external validity.

Key steps:
1. Download international index data via Yahoo Finance
2. Replicate A2 methodology using international passive fund flows
3. Cross-country heterogeneity: beta(c) ~ Passive_Share + Index_Concentration

Key outputs: International flow elasticities, cross-country comparison
"""
import logging

import numpy as np
import pandas as pd

from jappelli_experiments.config import INTERNATIONAL_INDICES, NW_LAGS_MONTHLY
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig

logger = logging.getLogger(__name__)
setup_plots()


def download_international_indices(start="2000-01-01", end="2024-12-31"):
    """
    Download international equity indices via yfinance.

    Returns
    -------
    DataFrame with monthly returns for each country.
    """
    import yfinance as yf

    results = {}
    for country, ticker in INTERNATIONAL_INDICES.items():
        try:
            data = yf.download(ticker, start=start, end=end, interval="1mo",
                                progress=False)
            if len(data) > 0:
                data = data[["Adj Close"]].copy()
                data.columns = [f"{country}_price"]
                data[f"{country}_ret"] = data[f"{country}_price"].pct_change()
                results[country] = data
                logger.info(f"  {country} ({ticker}): {len(data)} months")
        except Exception as e:
            logger.warning(f"  {country} ({ticker}) download failed: {e}")

    if results:
        merged = pd.concat([v for v in results.values()], axis=1)
        merged.index = pd.to_datetime(merged.index) + pd.offsets.MonthEnd(0)
        return merged

    return pd.DataFrame()


def international_flow_elasticity(index_returns, flow_proxy, country,
                                    nw_lags=NW_LAGS_MONTHLY):
    """
    Estimate flow elasticity for an international market.

    Parameters
    ----------
    index_returns : Series
        Monthly index returns.
    flow_proxy : Series
        Passive fund flow proxy (e.g., ETF flows, ICI aggregate data).
    country : str
        Country name.

    Returns
    -------
    dict with elasticity estimates.
    """
    df = pd.DataFrame({
        "ret": index_returns,
        "flow": flow_proxy,
    }).dropna()

    if len(df) < 30:
        return {"country": country, "n_obs": len(df), "note": "Insufficient data"}

    result = ols_nw(df["ret"].values, df[["flow"]].values,
                     max_lag=nw_lags, add_constant=True)

    return {
        "country": country,
        "beta": result.params[1],
        "se": result.bse[1],
        "t_stat": result.tvalues[1],
        "p_value": result.pvalues[1],
        "r2": result.rsquared,
        "n_obs": int(result.nobs),
    }


def run_e1(us_results=None, save_outputs=True):
    """
    Run the full E1 experiment.

    Parameters
    ----------
    us_results : dict
        US results from A2 for comparison.
    save_outputs : bool
        Save results.

    Returns
    -------
    dict with international results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT E1: International Validation")
    logger.info("=" * 60)

    results = {}

    # --- Download international data ---
    logger.info("Step 1: Downloading international indices...")
    intl_data = download_international_indices()
    results["index_data"] = intl_data

    if len(intl_data) == 0:
        logger.error("No international data available.")
        return results

    # --- Descriptive statistics ---
    logger.info("Step 2: Descriptive statistics...")
    ret_cols = [c for c in intl_data.columns if c.endswith("_ret")]
    desc = intl_data[ret_cols].describe()
    results["descriptives"] = desc
    logger.info(f"\n{desc.to_string()}")

    # --- Correlations ---
    logger.info("Step 3: Cross-country correlations...")
    corr = intl_data[ret_cols].corr()
    results["correlations"] = corr

    # Note: Full flow elasticity requires international fund flow data
    # which may be limited. Flag this for the user.
    logger.info("NOTE: International flow elasticity requires country-specific "
                "passive fund flow data (ICI Global, EFAMA). "
                "Currently only index return analysis available.")

    # --- Save ---
    if save_outputs:
        to_csv(desc, "e1_international_descriptives.csv")
        to_csv(corr, "e1_international_correlations.csv")

        # Figure: international indices
        fig, ax = plt.subplots(figsize=(12, 6))
        price_cols = [c for c in intl_data.columns if c.endswith("_price")]
        for col in price_cols:
            country = col.replace("_price", "")
            normalized = intl_data[col] / intl_data[col].dropna().iloc[0] * 100
            ax.plot(normalized.index, normalized.values, label=country, linewidth=1.5)
        ax.set_ylabel("Normalized Price (100 = start)")
        ax.set_title("International Equity Indices")
        ax.legend()
        save_fig(fig, "e1_international_indices.pdf")

    logger.info("=" * 60)
    logger.info("E1 COMPLETE")
    logger.info("=" * 60)

    return results
