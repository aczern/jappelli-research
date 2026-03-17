"""
Experiment B2: Heterogeneous Effects by Firm Characteristics

Tests how the static ownership effect varies across firm characteristics:
high-beta, illiquid, high idiosyncratic volatility, index-weight stocks.

Key steps:
1. Compute firm characteristics: beta, Amihud illiquidity, idio vol, index weight
2. Interaction models: SO * High-Beta, SO * Low-Liq, etc.
3. Double sorts: characteristic quartile x SO quintile
4. Causal forest for non-parametric heterogeneity
5. Multiple testing correction (Benjamini-Hochberg)

Key outputs: Heterogeneity patterns, interaction coefficients
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from jappelli_experiments.config import ROLLING_BETA_WINDOW, NW_LAGS_MONTHLY
from jappelli_experiments.shared.fama_macbeth import fama_macbeth
from jappelli_experiments.shared.portfolio_sorts import double_sort, portfolio_alpha
from jappelli_experiments.shared.statistical_tests import benjamini_hochberg
from jappelli_experiments.shared.newey_west import ols_nw
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cleaning import winsorize_panel

logger = logging.getLogger(__name__)
setup_plots()


def compute_firm_characteristics(stock_panel, ff_factors):
    """
    Compute firm-level characteristics for heterogeneity analysis.

    Parameters
    ----------
    stock_panel : DataFrame
        Stock-month panel with ret, me, vol.
    ff_factors : DataFrame
        FF factors for beta/idiosyncratic vol estimation.

    Returns
    -------
    DataFrame with permno, date, and computed characteristics.
    """
    df = stock_panel.copy()

    # --- Market beta (60-month rolling) ---
    if ff_factors is not None and "Mkt-RF" in ff_factors.columns:
        # Merge FF factors
        ff = ff_factors.copy()
        if "date" in ff.columns:
            ff = ff.set_index("date")

        df_with_ff = pd.merge(df, ff[["Mkt-RF", "SMB", "HML", "RF"]],
                               left_on="date", right_index=True, how="left")

        if "ret" in df_with_ff.columns and "RF" in df_with_ff.columns:
            df_with_ff["excess_ret"] = df_with_ff["ret"] - df_with_ff["RF"]

            # Rolling beta
            def _rolling_beta(group):
                if len(group) < 24:
                    return pd.Series(np.nan, index=group.index)
                return group["excess_ret"].rolling(ROLLING_BETA_WINDOW, min_periods=24).apply(
                    lambda y: np.polyfit(
                        group.loc[y.index, "Mkt-RF"].values, y.values, 1
                    )[0] if len(y) >= 24 else np.nan,
                    raw=False
                )

            # Simplified: use expanding regression per stock
            betas = []
            for permno, grp in df_with_ff.groupby("permno"):
                grp = grp.sort_values("date")
                if len(grp) < 24:
                    continue
                er = grp["excess_ret"].values
                mkt = grp["Mkt-RF"].values

                beta_series = pd.Series(np.nan, index=grp.index)
                for i in range(ROLLING_BETA_WINDOW, len(grp)):
                    start = max(0, i - ROLLING_BETA_WINDOW)
                    y = er[start:i]
                    x = mkt[start:i]
                    valid = ~(np.isnan(y) | np.isnan(x))
                    if valid.sum() >= 24:
                        beta_series.iloc[i] = np.polyfit(x[valid], y[valid], 1)[0]

                betas.append(pd.DataFrame({
                    "permno": permno,
                    "date": grp["date"].values,
                    "beta": beta_series.values,
                }))

            if betas:
                beta_df = pd.concat(betas, ignore_index=True)
                df = pd.merge(df, beta_df, on=["permno", "date"], how="left")

            # Idiosyncratic volatility (residual from FF3)
            idio_vols = []
            for permno, grp in df_with_ff.groupby("permno"):
                grp = grp.sort_values("date")
                if len(grp) < 24:
                    continue
                er = grp["excess_ret"].values
                X = grp[["Mkt-RF", "SMB", "HML"]].values

                ivol_series = pd.Series(np.nan, index=grp.index)
                for i in range(ROLLING_BETA_WINDOW, len(grp)):
                    start = max(0, i - ROLLING_BETA_WINDOW)
                    y = er[start:i]
                    x = X[start:i]
                    valid = ~np.any(np.isnan(np.column_stack([y, x])), axis=1)
                    if valid.sum() >= 24:
                        try:
                            res = sm.OLS(y[valid], sm.add_constant(x[valid])).fit()
                            ivol_series.iloc[i] = np.std(res.resid)
                        except Exception:
                            pass

                idio_vols.append(pd.DataFrame({
                    "permno": permno,
                    "date": grp["date"].values,
                    "idio_vol": ivol_series.values,
                }))

            if idio_vols:
                ivol_df = pd.concat(idio_vols, ignore_index=True)
                df = pd.merge(df, ivol_df, on=["permno", "date"], how="left")

    # --- Amihud illiquidity ---
    # Requires daily data; use monthly proxy: |ret| / volume
    if "ret" in df.columns and "vol" in df.columns:
        df["amihud"] = df["ret"].abs() / (df["vol"] * df["prc"].abs() / 1e6 + 1e-8)
        df["amihud"] = df["amihud"].clip(upper=df["amihud"].quantile(0.99))

    logger.info(f"Characteristics computed for {df['permno'].nunique():,} stocks")
    return df


def interaction_models(stock_panel, characteristics):
    """
    Run Fama-MacBeth with SO x characteristic interactions.

    Parameters
    ----------
    stock_panel : DataFrame
        Panel with ret, SO_lag, and characteristics.
    characteristics : list of str
        Characteristic column names to interact with SO.

    Returns
    -------
    dict mapping characteristic -> FM results
    """
    df = stock_panel.copy()
    df["SO_lag"] = df.groupby("permno")["static_ownership"].shift(1)

    results = {}
    p_values = []

    for char in characteristics:
        if char not in df.columns:
            continue

        # Create high/low indicator (median split)
        df[f"high_{char}"] = (
            df.groupby("date")[char].transform(
                lambda x: (x >= x.median()).astype(float)
            )
        )
        df[f"SO_x_{char}"] = df["SO_lag"] * df[f"high_{char}"]

        try:
            fm = fama_macbeth(
                df, "date", "ret",
                ["SO_lag", f"high_{char}", f"SO_x_{char}"]
            )
            results[char] = fm
            p_val = fm.p_values[f"SO_x_{char}"]
            p_values.append({"characteristic": char, "p_value": p_val})
            logger.info(f"  {char}: interaction beta = {fm.mean_coefs[f'SO_x_{char}']:.4f} "
                        f"(t = {fm.t_stats[f'SO_x_{char}']:.3f})")
        except Exception as e:
            logger.warning(f"  {char} interaction failed: {e}")

    # Multiple testing correction
    if p_values:
        p_df = pd.DataFrame(p_values)
        bh = benjamini_hochberg(p_df["p_value"].values)
        results["multiple_testing"] = pd.concat([p_df, bh], axis=1)

    return results


def run_b2(stock_panel, ff_factors=None, save_outputs=True):
    """
    Run the full B2 experiment.

    Parameters
    ----------
    stock_panel : DataFrame
        Stock-month panel with static_ownership.
    ff_factors : DataFrame
        Fama-French factors.
    save_outputs : bool
        Save results to disk.

    Returns
    -------
    dict with all B2 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT B2: Heterogeneous Effects")
    logger.info("=" * 60)

    results = {}

    # --- Step 1: Compute characteristics ---
    logger.info("Step 1: Computing firm characteristics...")
    panel = compute_firm_characteristics(stock_panel, ff_factors)

    # --- Step 2: Interaction models ---
    logger.info("Step 2: Interaction models...")
    chars = ["beta", "amihud", "idio_vol", "log_me"]
    available_chars = [c for c in chars if c in panel.columns]

    if available_chars:
        interaction_results = interaction_models(panel, available_chars)
        results["interactions"] = interaction_results

    # --- Step 3: Double sorts ---
    logger.info("Step 3: Double sorts...")
    if "static_ownership" in panel.columns:
        panel["SO_lag"] = panel.groupby("permno")["static_ownership"].shift(1)

        for char in available_chars:
            try:
                ds = double_sort(panel, "date", "ret", char, "SO_lag",
                                  n1=4, n2=5, conditional=True)
                results[f"double_sort_{char}"] = ds
            except Exception as e:
                logger.warning(f"  Double sort on {char} failed: {e}")

    # --- Step 4: Causal forest (optional) ---
    logger.info("Step 4: Causal forest (if econml available)...")
    try:
        from econml.dml import CausalForestDML

        cf_panel = panel.dropna(subset=["ret", "SO_lag"] + available_chars)
        if len(cf_panel) > 1000:
            Y = cf_panel["ret"].values
            T = cf_panel["SO_lag"].values
            X = cf_panel[available_chars].values
            W = cf_panel[["log_me"]].values if "log_me" in cf_panel.columns else None

            cf = CausalForestDML(n_estimators=200, random_state=42)
            cf.fit(Y, T, X=X, W=W)

            te = cf.effect(X)
            results["causal_forest"] = {
                "mean_te": np.mean(te),
                "std_te": np.std(te),
                "median_te": np.median(te),
            }
            logger.info(f"  Causal forest: mean TE = {np.mean(te):.4f}")
    except ImportError:
        logger.info("  econml not available; skipping causal forest")
    except Exception as e:
        logger.warning(f"  Causal forest failed: {e}")

    # --- Save ---
    if save_outputs and "interactions" in results:
        rows = []
        for char, fm in results["interactions"].items():
            if hasattr(fm, "summary"):
                summary = fm.summary()
                for var in summary.index:
                    rows.append({
                        "Characteristic": char,
                        "Variable": var,
                        "Coef": summary.loc[var, "coef"],
                        "SE": summary.loc[var, "se_nw"],
                        "t": summary.loc[var, "t_stat"],
                    })
        if rows:
            to_csv(pd.DataFrame(rows), "b2_heterogeneous_effects.csv")

    logger.info("=" * 60)
    logger.info("B2 COMPLETE")
    logger.info("=" * 60)

    return results
