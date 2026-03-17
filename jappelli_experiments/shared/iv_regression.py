"""
Two-stage least squares (2SLS) IV regression with diagnostics.

Wraps linearmodels.iv for panel and cross-section IV estimation.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS


def iv_2sls(data, y_col, endog_cols, exog_cols, instrument_cols,
            add_constant=True):
    """
    Run 2SLS IV regression.

    Parameters
    ----------
    data : DataFrame
        Regression data.
    y_col : str
        Dependent variable.
    endog_cols : list of str
        Endogenous regressors.
    exog_cols : list of str
        Exogenous controls (included instruments).
    instrument_cols : list of str
        Excluded instruments.
    add_constant : bool
        Add intercept to exogenous variables.

    Returns
    -------
    dict with: results, first_stage, diagnostics
    """
    df = data[[y_col] + endog_cols + exog_cols + instrument_cols].dropna()

    dependent = df[y_col]
    endog = df[endog_cols]
    exog = df[exog_cols]
    instruments = df[instrument_cols]

    if add_constant:
        exog = sm.add_constant(exog)

    model = IV2SLS(dependent, exog, endog, instruments)
    results = model.fit(cov_type="robust")

    # First-stage diagnostics
    first_stage_results = {}
    for col in endog_cols:
        X_first = pd.concat([exog, df[instrument_cols]], axis=1)
        y_first = df[col]
        fs_model = sm.OLS(y_first, X_first, missing="drop").fit()
        first_stage_results[col] = {
            "f_stat": fs_model.fvalue,
            "f_pvalue": fs_model.f_pvalue,
            "r2": fs_model.rsquared,
            "partial_r2": _partial_r2(df, col, instrument_cols, exog_cols),
        }

    diagnostics = {
        "first_stage": first_stage_results,
        "n_obs": results.nobs,
        "n_instruments": len(instrument_cols),
        "n_endogenous": len(endog_cols),
    }

    # Over-identification test (Hansen J) if overidentified
    if len(instrument_cols) > len(endog_cols):
        try:
            diagnostics["hansen_j"] = float(results.j_stat.stat)
            diagnostics["hansen_j_pvalue"] = float(results.j_stat.pval)
        except Exception:
            diagnostics["hansen_j"] = np.nan
            diagnostics["hansen_j_pvalue"] = np.nan

    return {
        "results": results,
        "first_stage": first_stage_results,
        "diagnostics": diagnostics,
    }


def _partial_r2(data, endog_col, instruments, controls):
    """Compute partial R-squared of instruments in first stage."""
    df = data[[endog_col] + instruments + controls].dropna()

    # Restricted model (only controls)
    X_r = sm.add_constant(df[controls].values) if controls else sm.add_constant(np.ones((len(df), 1)))
    res_r = sm.OLS(df[endog_col].values, X_r).fit()

    # Unrestricted model (controls + instruments)
    X_u = sm.add_constant(df[controls + instruments].values)
    res_u = sm.OLS(df[endog_col].values, X_u).fit()

    ssr_r = res_r.ssr
    ssr_u = res_u.ssr
    return (ssr_r - ssr_u) / ssr_r


def weak_instrument_test(first_stage_f, n_endogenous=1, significance=0.05):
    """
    Check against Stock-Yogo (2005) critical values for weak instruments.

    Returns
    -------
    dict with passed (bool), f_stat, critical_value
    """
    # Stock-Yogo critical values for maximal IV size (5% significance)
    # For 1 endogenous regressor, different numbers of instruments
    sy_critical = {1: 16.38, 2: 19.93, 3: 22.30}

    cv = sy_critical.get(n_endogenous, 10.0)  # Default rule of thumb

    return {
        "passed": first_stage_f > cv,
        "f_stat": first_stage_f,
        "critical_value": cv,
        "rule_of_thumb_passed": first_stage_f > 10.0,
    }
