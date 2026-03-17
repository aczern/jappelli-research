"""
VAR and structural VAR models with impulse response functions.

Wraps statsmodels VAR with additional diagnostics and Cholesky identification.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as StatsVAR
from jappelli_experiments.config import VAR_MAX_LAGS, IRF_HORIZON


def estimate_var(data, var_names, max_lags=VAR_MAX_LAGS, ic="aic"):
    """
    Estimate a reduced-form VAR with automatic lag selection.

    Parameters
    ----------
    data : DataFrame
        DataFrame with columns in var_names. Must be stationary.
    var_names : list of str
        Variable names (column order determines Cholesky ordering).
    max_lags : int
        Maximum lags to consider for information criterion.
    ic : str
        Information criterion: 'aic', 'bic', 'hqic', 'fpe'.

    Returns
    -------
    dict with keys: model, results, selected_lags, ic_table
    """
    df = data[var_names].dropna()
    model = StatsVAR(df)

    # Lag selection
    lag_order = model.select_order(maxlags=min(max_lags, len(df) // 3))
    selected = getattr(lag_order, ic)
    ic_table = lag_order.summary()

    results = model.fit(maxlags=selected)

    return {
        "model": model,
        "results": results,
        "selected_lags": selected,
        "ic_table": ic_table,
    }


def impulse_responses(var_results, impulse, response, periods=IRF_HORIZON,
                      orth=True, ci=0.95):
    """
    Compute impulse response functions from estimated VAR.

    Parameters
    ----------
    var_results : VARResults
        Fitted VAR model.
    impulse : str
        Name of the shock variable.
    response : str
        Name of the response variable.
    periods : int
        Horizon for IRFs.
    orth : bool
        Use orthogonalized (Cholesky) IRFs.
    ci : float
        Confidence interval level for bootstrap bands.

    Returns
    -------
    DataFrame with columns: irf, lower, upper
    """
    irf_obj = var_results.irf(periods)

    # Get variable indices
    names = var_results.names
    imp_idx = names.index(impulse)
    resp_idx = names.index(response)

    irf_values = irf_obj.orth_irfs[:, resp_idx, imp_idx] if orth else irf_obj.irfs[:, resp_idx, imp_idx]

    # Bootstrap confidence intervals
    try:
        irf_err = var_results.irf_errband_mc(
            orth=orth, repl=1000, steps=periods,
            signif=1 - ci, seed=42
        )
        lower = irf_err[:, resp_idx, imp_idx, 0]
        upper = irf_err[:, resp_idx, imp_idx, 1]
    except Exception:
        # Fallback: no confidence bands
        lower = np.full(periods + 1, np.nan)
        upper = np.full(periods + 1, np.nan)

    return pd.DataFrame({
        "irf": irf_values,
        "lower": lower,
        "upper": upper,
    }, index=range(periods + 1))


def forecast_error_variance_decomposition(var_results, periods=IRF_HORIZON):
    """
    Compute forecast error variance decomposition (FEVD).

    Returns
    -------
    dict mapping response variable -> DataFrame of FEVD contributions
    """
    fevd = var_results.fevd(periods)
    decomp = {}
    for i, name in enumerate(var_results.names):
        decomp[name] = pd.DataFrame(
            fevd.decomp[i],
            columns=var_results.names,
            index=range(periods + 1),
        )
    return decomp


def granger_causality_tests(var_results):
    """
    Run pairwise Granger causality tests for all variables in the VAR.

    Returns
    -------
    DataFrame with columns: cause, effect, f_stat, p_value
    """
    names = var_results.names
    results_list = []

    for cause in names:
        for effect in names:
            if cause == effect:
                continue
            try:
                test = var_results.test_causality(effect, [cause], kind="f")
                results_list.append({
                    "cause": cause,
                    "effect": effect,
                    "f_stat": test.test_statistic,
                    "p_value": test.pvalue,
                })
            except Exception:
                continue

    return pd.DataFrame(results_list)
