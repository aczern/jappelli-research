"""
Jorda (2005) Local Projection Impulse Response Functions.

Estimates IRFs by running direct regressions at each horizon h:
    y_{t+h} = alpha(h) + beta(h) * shock_t + gamma(h) * controls_t + eps
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from jappelli_experiments.config import NW_LAGS_MONTHLY, IRF_HORIZON


def local_projection_irf(data, y_col, shock_col, control_cols=None,
                         horizons=IRF_HORIZON, nw_lags=NW_LAGS_MONTHLY,
                         add_constant=True):
    """
    Estimate local projection IRFs.

    Parameters
    ----------
    data : DataFrame
        Time series data (must be sorted by time, no gaps).
    y_col : str
        Outcome variable.
    shock_col : str
        Shock / treatment variable.
    control_cols : list of str or None
        Control variables (lags of y and shock are typically included).
    horizons : int
        Maximum horizon.
    nw_lags : int
        Newey-West lag truncation.
    add_constant : bool
        Add intercept.

    Returns
    -------
    DataFrame with columns: horizon, beta, se, t_stat, p_value, ci_lower, ci_upper, n_obs
    """
    if control_cols is None:
        control_cols = []

    results = []
    n_total = len(data)

    for h in range(horizons + 1):
        # Forward y by h periods
        df = data.copy()
        df["y_forward"] = df[y_col].shift(-h)
        df = df.dropna(subset=["y_forward", shock_col] + control_cols)

        if len(df) < 20:
            results.append({
                "horizon": h, "beta": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "n_obs": 0,
            })
            continue

        X_cols = [shock_col] + control_cols
        X = df[X_cols].values
        if add_constant:
            X = sm.add_constant(X)
        y = df["y_forward"].values

        model = sm.OLS(y, X, missing="drop")
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

        # shock_col coefficient is at index 1 (after constant) or 0
        idx = 1 if add_constant else 0
        beta = res.params[idx]
        se = res.bse[idx]
        t_stat = res.tvalues[idx]
        p_value = res.pvalues[idx]
        ci = res.conf_int()[idx]

        results.append({
            "horizon": h,
            "beta": beta,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "n_obs": int(res.nobs),
        })

    return pd.DataFrame(results)


def lp_regime_dependent(data, y_col, shock_col, regime_col, control_cols=None,
                        horizons=IRF_HORIZON, nw_lags=NW_LAGS_MONTHLY):
    """
    Regime-dependent local projections using interaction terms.

    y_{t+h} = alpha + beta_low(h)*shock*I(regime=low) + beta_high(h)*shock*I(regime=high) + controls + eps

    Parameters
    ----------
    regime_col : str
        Binary indicator column (1 = high regime).

    Returns
    -------
    DataFrame with columns for both regime coefficients.
    """
    if control_cols is None:
        control_cols = []

    results = []

    for h in range(horizons + 1):
        df = data.copy()
        df["y_forward"] = df[y_col].shift(-h)
        df["shock_high"] = df[shock_col] * df[regime_col]
        df["shock_low"] = df[shock_col] * (1 - df[regime_col])
        df = df.dropna(subset=["y_forward", "shock_high", "shock_low"] + control_cols)

        if len(df) < 20:
            results.append({"horizon": h})
            continue

        X_cols = ["shock_low", "shock_high"] + control_cols
        X = sm.add_constant(df[X_cols].values)
        y = df["y_forward"].values

        res = sm.OLS(y, X, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": nw_lags}
        )

        results.append({
            "horizon": h,
            "beta_low": res.params[1],
            "se_low": res.bse[1],
            "beta_high": res.params[2],
            "se_high": res.bse[2],
            "n_obs": int(res.nobs),
        })

    return pd.DataFrame(results)
