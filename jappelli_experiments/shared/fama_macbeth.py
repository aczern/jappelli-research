"""
Fama-MacBeth (1973) cross-sectional regressions with Newey-West correction.

Usage:
    results = fama_macbeth(panel, 'date', 'ret', ['SO_lag', 'log_me', 'bm'])
    print(results.summary())
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from jappelli_experiments.config import FM_MIN_OBS, NW_LAGS_MONTHLY
from jappelli_experiments.shared.newey_west import newey_west_se


class FamaMacBethResult:
    """Container for Fama-MacBeth regression output."""

    def __init__(self, coef_ts, mean_coefs, nw_se, t_stats, p_values,
                 n_periods, avg_n_stocks, r2_ts):
        self.coef_ts = coef_ts          # DataFrame: time series of coefficients
        self.mean_coefs = mean_coefs    # Series: time-series average coefficients
        self.nw_se = nw_se              # Series: Newey-West standard errors
        self.t_stats = t_stats          # Series: t-statistics
        self.p_values = p_values        # Series: p-values
        self.n_periods = n_periods
        self.avg_n_stocks = avg_n_stocks
        self.r2_ts = r2_ts              # Series: R-squared per period

    def summary(self):
        """Return a formatted summary DataFrame."""
        df = pd.DataFrame({
            "coef": self.mean_coefs,
            "se_nw": self.nw_se,
            "t_stat": self.t_stats,
            "p_value": self.p_values,
        })
        df.attrs["n_periods"] = self.n_periods
        df.attrs["avg_n_stocks"] = self.avg_n_stocks
        df.attrs["avg_r2"] = self.r2_ts.mean()
        return df


def fama_macbeth(panel, time_col, y_col, x_cols, add_constant=True,
                 nw_lags=NW_LAGS_MONTHLY, min_obs=FM_MIN_OBS):
    """
    Run Fama-MacBeth cross-sectional regressions.

    Parameters
    ----------
    panel : DataFrame
        Panel data with entity and time dimensions.
    time_col : str
        Column identifying time periods.
    y_col : str
        Dependent variable column.
    x_cols : list of str
        Independent variable columns.
    add_constant : bool
        Whether to add an intercept.
    nw_lags : int
        Newey-West lag truncation.
    min_obs : int
        Minimum cross-section size per period.

    Returns
    -------
    FamaMacBethResult
    """
    df = panel[[time_col, y_col] + x_cols].dropna()
    periods = df[time_col].unique()

    coef_list = []
    r2_list = []
    n_stocks_list = []

    for t in sorted(periods):
        cross = df[df[time_col] == t]
        if len(cross) < min_obs:
            continue

        y = cross[y_col].values
        X = cross[x_cols].values
        if add_constant:
            X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            continue

        names = ["const"] + x_cols if add_constant else x_cols
        coef_list.append(pd.Series(model.params, index=names, name=t))
        r2_list.append(model.rsquared)
        n_stocks_list.append(len(cross))

    if not coef_list:
        raise ValueError("No valid cross-sections found.")

    coef_ts = pd.DataFrame(coef_list)
    mean_coefs = coef_ts.mean()

    # Newey-West standard errors on the time series of coefficients
    nw_se = pd.Series(
        {col: newey_west_se(coef_ts[col].values, nw_lags) for col in coef_ts.columns},
    )
    t_stats = mean_coefs / nw_se
    from scipy import stats as sp_stats
    p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), df=len(coef_ts) - 1))
    p_values = pd.Series(p_values, index=t_stats.index)

    return FamaMacBethResult(
        coef_ts=coef_ts,
        mean_coefs=mean_coefs,
        nw_se=nw_se,
        t_stats=t_stats,
        p_values=p_values,
        n_periods=len(coef_ts),
        avg_n_stocks=np.mean(n_stocks_list),
        r2_ts=pd.Series(r2_list, index=coef_ts.index),
    )
