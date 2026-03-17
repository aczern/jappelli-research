"""
Cross-experiment dependency tracking and consistency validation.

Tracks which experiment outputs feed into which downstream experiments,
and validates that shared variables (theta_t, V_t, fund classification)
are identical across all consumers.
"""
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from jappelli_experiments.config import INTERMEDIATE_DIR, LOG_DIR

logger = logging.getLogger(__name__)

# ── Dependency graph ──
DEPENDENCY_GRAPH = {
    "A1": {
        "inputs": ["aggregate_monthly_panel"],
        "outputs": ["V_t", "PDV_t", "theta_t"],
        "feeds": ["C1", "E2", "E4"],
    },
    "A2": {
        "inputs": ["aggregate_monthly_panel", "sp500_adddrop"],
        "outputs": ["flow_elasticity", "flow_definitions"],
        "feeds": ["C2", "D1", "E1", "E3"],
    },
    "B1": {
        "inputs": ["stock_month_panel", "fund_holdings"],
        "outputs": ["SO_it", "ownership_return_link"],
        "feeds": ["B2", "E4"],
    },
    "B2": {
        "inputs": ["B1.SO_it", "firm_characteristics"],
        "outputs": ["heterogeneous_effects"],
        "feeds": [],
    },
    "C1": {
        "inputs": ["A1.V_t", "A1.theta_t", "sharpe_t"],
        "outputs": ["time_varying_beta", "amplification_coef"],
        "feeds": ["D2", "E2"],
    },
    "C2": {
        "inputs": ["A2.flow_definitions", "sharpe_t", "vix"],
        "outputs": ["irf_estimates", "half_life"],
        "feeds": ["E2"],
    },
    "D1": {
        "inputs": ["A2.flow_definitions", "fund_classification"],
        "outputs": ["static_elasticity", "dynamic_elasticity"],
        "feeds": ["D2"],
    },
    "D2": {
        "inputs": ["D1.static_elasticity", "C1.amplification_coef", "vix"],
        "outputs": ["state_dependent_elasticity", "spillover_estimates"],
        "feeds": [],
    },
    "E1": {
        "inputs": ["international_indices", "A2.methodology"],
        "outputs": ["international_results"],
        "feeds": [],
    },
    "E2": {
        "inputs": ["A1.V_t", "C1.amplification_coef", "daily_returns"],
        "outputs": ["stress_test_results"],
        "feeds": [],
    },
    "E3": {
        "inputs": ["A2.fund_classification", "fund_holdings"],
        "outputs": ["rebalancing_elasticity"],
        "feeds": [],
    },
    "E4": {
        "inputs": ["B1.SO_it", "A1.V_t", "ff_factors"],
        "outputs": ["factor_bubble_estimates"],
        "feeds": [],
    },
}


def _hash_series(s):
    """Compute hash of a pandas Series for consistency checking."""
    arr = np.asarray(s, dtype=float)
    arr = arr[~np.isnan(arr)]
    return hashlib.md5(arr.tobytes()).hexdigest()


class ConnectionMapper:
    """Track and validate cross-experiment dependencies."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else INTERMEDIATE_DIR
        self.registry = {}  # {variable_name: {hash, experiment, timestamp}}
        self._log_path = LOG_DIR / "connection_checks.log"

    def register_output(self, experiment, variable_name, data):
        """
        Register an experiment output for downstream consumption.

        Parameters
        ----------
        experiment : str
            Experiment ID (e.g., 'A1').
        variable_name : str
            Variable name (e.g., 'V_t', 'theta_t').
        data : Series or DataFrame
            The output data.
        """
        if isinstance(data, pd.DataFrame):
            h = hashlib.md5(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()
        else:
            h = _hash_series(data)

        self.registry[variable_name] = {
            "hash": h,
            "experiment": experiment,
            "timestamp": datetime.now().isoformat(),
            "shape": data.shape if hasattr(data, "shape") else len(data),
        }

        # Save to disk
        out_path = self.cache_dir / f"{experiment}_{variable_name}.parquet"
        if isinstance(data, pd.Series):
            data.to_frame(variable_name).to_parquet(out_path)
        else:
            data.to_parquet(out_path)

        logger.info(f"Registered {experiment}.{variable_name} (hash={h[:8]})")

    def validate_input(self, consumer_experiment, variable_name, data):
        """
        Validate that the data a consumer is using matches the registered output.

        Returns
        -------
        bool : True if consistent, False if mismatch.
        """
        if variable_name not in self.registry:
            logger.warning(f"{variable_name} not registered. Cannot validate.")
            return True  # Cannot check

        if isinstance(data, pd.DataFrame):
            h = hashlib.md5(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()
        else:
            h = _hash_series(data)

        expected = self.registry[variable_name]["hash"]
        producer = self.registry[variable_name]["experiment"]

        if h != expected:
            msg = (
                f"CONSISTENCY FAILURE: {consumer_experiment} using "
                f"{variable_name} (hash={h[:8]}) does not match "
                f"{producer} output (hash={expected[:8]})"
            )
            logger.error(msg)
            self._log_check(msg)
            return False

        msg = f"OK: {consumer_experiment} <- {producer}.{variable_name}"
        logger.info(msg)
        self._log_check(msg)
        return True

    def run_all_checks(self):
        """
        Run consistency checks across all registered variables.

        Returns
        -------
        dict mapping variable -> bool (pass/fail)
        """
        results = {}
        for var_name, info in self.registry.items():
            # Load from disk and verify hash
            path = self.cache_dir / f"{info['experiment']}_{var_name}.parquet"
            if path.exists():
                loaded = pd.read_parquet(path)
                h = hashlib.md5(pd.util.hash_pandas_object(loaded).values.tobytes()).hexdigest()
                results[var_name] = h == info["hash"]
            else:
                results[var_name] = None  # Cannot check
        return results

    def get_dependency_chain(self, experiment):
        """Get all upstream dependencies for an experiment."""
        if experiment not in DEPENDENCY_GRAPH:
            return []
        inputs = DEPENDENCY_GRAPH[experiment]["inputs"]
        upstream = []
        for inp in inputs:
            if "." in inp:
                exp, _ = inp.split(".", 1)
                upstream.append(exp)
                upstream.extend(self.get_dependency_chain(exp))
        return list(set(upstream))

    def _log_check(self, msg):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")

    def summary(self):
        """Print summary of registered outputs."""
        rows = []
        for var_name, info in self.registry.items():
            rows.append({
                "variable": var_name,
                "producer": info["experiment"],
                "hash": info["hash"][:12],
                "shape": str(info["shape"]),
                "timestamp": info["timestamp"],
            })
        return pd.DataFrame(rows)
