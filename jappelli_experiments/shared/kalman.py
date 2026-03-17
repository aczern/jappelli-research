"""
Kalman filter / state-space models for time-varying parameter estimation.

Used in C1 for estimating time-varying beta in V_t ~ theta_t regression.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel


class TimeVaryingParameterModel(MLEModel):
    """
    State-space model with time-varying coefficients.

    Observation equation:
        y_t = X_t * beta_t + eps_t,   eps_t ~ N(0, sigma2_obs)

    State transition:
        beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, sigma2_state)

    Parameters to estimate: sigma2_obs, sigma2_state (for each coefficient)
    """

    def __init__(self, endog, exog, **kwargs):
        k_states = exog.shape[1]
        super().__init__(endog, k_states=k_states, k_posdef=k_states, **kwargs)
        self.exog_data = np.asarray(exog)

        # Initialize state space matrices
        self["transition"] = np.eye(k_states)
        self["selection"] = np.eye(k_states)

        # Initial state: diffuse prior
        self.ssm.initialization = None
        self.initialize_approximate_diffuse(1e6)

    @property
    def param_names(self):
        names = ["sigma2_obs"]
        for i in range(self.k_states):
            names.append(f"sigma2_state_{i}")
        return names

    @property
    def start_params(self):
        return np.ones(1 + self.k_states) * 0.1

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        # Observation variance
        self["obs_cov", 0, 0] = params[0]

        # State innovation variance
        for i in range(self.k_states):
            self["state_cov", i, i] = params[1 + i]

        # Time-varying design matrix
        for t in range(self.nobs):
            self["design", 0, :, t] = self.exog_data[t]

    def transform_params(self, unconstrained):
        return unconstrained ** 2

    def untransform_params(self, constrained):
        return constrained ** 0.5


def tvp_regression(y, X, col_names=None):
    """
    Estimate time-varying parameter regression via Kalman filter.

    Parameters
    ----------
    y : array-like
        Dependent variable (T,).
    X : array-like
        Regressors (T, k). Include constant column if desired.
    col_names : list of str or None
        Names for the state variables.

    Returns
    -------
    dict with:
        filtered_states : DataFrame (T x k) of filtered beta_t
        smoothed_states : DataFrame (T x k) of smoothed beta_t
        params : estimated variance parameters
        loglik : log-likelihood
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    k = X.shape[1]
    if col_names is None:
        col_names = [f"beta_{i}" for i in range(k)]

    model = TimeVaryingParameterModel(y, X)
    results = model.fit(disp=False, maxiter=500)

    filtered = pd.DataFrame(
        results.filtered_state.T,
        columns=col_names,
    )
    smoothed = pd.DataFrame(
        results.smoothed_state.T,
        columns=col_names,
    )

    return {
        "filtered_states": filtered,
        "smoothed_states": smoothed,
        "params": results.params,
        "loglik": results.llf,
        "model_results": results,
    }
