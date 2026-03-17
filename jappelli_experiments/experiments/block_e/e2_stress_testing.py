"""
Experiment E2: Stress-Testing the Rational Bubble

Tests bubble behavior during market crises. Examines whether the
bubble component V_t declines more than fundamentals during stress events.

Key steps:
1. Identify crisis events (single-day returns < -5%)
2. Event study: actual V_t decline vs martingale-predicted decline
3. Amplification multiplier: total decline / fundamental decline
4. Structural simulation: GBM dividends + dynamic + static investors
5. Stress scenarios: mass redemption, credit crunch, VIX spike

Key outputs: Crisis amplification multiplier, simulation results
"""
import logging

import numpy as np
import pandas as pd

from jappelli_experiments.config import NW_LAGS_DAILY, RANDOM_SEED
from jappelli_experiments.shared.newey_west import ols_nw, nw_t_test
from jappelli_experiments.shared.connection_mapper import ConnectionMapper
from jappelli_experiments.shared.table_formatter import to_csv
from jappelli_experiments.shared.plot_config import setup_plots, save_fig
from jappelli_experiments.data.cache import load_cache

logger = logging.getLogger(__name__)
setup_plots()


def identify_crisis_events(returns, threshold=-0.05, col="vwretd"):
    """
    Identify crisis events: days with market return below threshold.

    Parameters
    ----------
    returns : DataFrame or Series
        Market returns (daily or monthly).
    threshold : float
        Return threshold (e.g., -0.05 for -5%).

    Returns
    -------
    DataFrame with crisis dates, returns, and severity rankings.
    """
    if isinstance(returns, pd.Series):
        ret = returns.dropna()
    else:
        ret = returns[col].dropna()

    crises = ret[ret < threshold].sort_values()

    result = pd.DataFrame({
        "date": crises.index,
        "return": crises.values,
        "rank": range(1, len(crises) + 1),
    })

    logger.info(f"Identified {len(result)} crisis events (threshold = {threshold:.1%})")
    return result


def crisis_bubble_response(V_t, crisis_dates, window=(-5, 20)):
    """
    Event study of bubble component response to crises.

    Parameters
    ----------
    V_t : Series
        Bubble component (should be at matching frequency).
    crisis_dates : list/array
        Crisis event dates.
    window : tuple
        (pre, post) periods around event.

    Returns
    -------
    dict with average response path and amplification metrics.
    """
    V = V_t.dropna()
    responses = []

    for date in crisis_dates:
        # Find nearest V_t observation
        idx = V.index.searchsorted(date)
        if idx < abs(window[0]) or idx + window[1] >= len(V):
            continue

        start = idx + window[0]
        end = idx + window[1] + 1
        v_window = V.iloc[start:end].values
        # Normalize to pre-event level
        v_norm = v_window / v_window[abs(window[0])] - 1 if v_window[abs(window[0])] != 0 else v_window
        responses.append(v_norm)

    if not responses:
        return {"n_events": 0}

    responses = np.array(responses)
    avg_response = np.nanmean(responses, axis=0)
    se_response = np.nanstd(responses, axis=0) / np.sqrt(len(responses))

    # Amplification: max decline / pre-event level
    max_decline = np.min(avg_response)
    recovery_idx = None
    for i in range(abs(window[0]), len(avg_response)):
        if avg_response[i] >= 0:
            recovery_idx = i - abs(window[0])
            break

    return {
        "n_events": len(responses),
        "avg_response": avg_response,
        "se_response": se_response,
        "max_decline": max_decline,
        "recovery_periods": recovery_idx,
        "time_axis": list(range(window[0], window[1] + 1)),
    }


def structural_simulation(n_periods=500, n_sims=1000, theta=0.15,
                           mu_d=0.02, sigma_d=0.04, r_f=0.03,
                           gamma=2.0, seed=RANDOM_SEED):
    """
    Structural simulation: GBM dividends + dynamic (CRRA) + static investors.

    Simulates the Jappelli model to generate bubble dynamics and
    test behavior under stress scenarios.

    Parameters
    ----------
    n_periods : int
        Simulation length (months).
    n_sims : int
        Number of Monte Carlo paths.
    theta : float
        Static investor equity allocation share.
    mu_d : float
        Dividend drift (annualized).
    sigma_d : float
        Dividend volatility (annualized).
    r_f : float
        Risk-free rate (annualized).
    gamma : float
        Risk aversion of dynamic investors.
    seed : int
        Random seed.

    Returns
    -------
    dict with simulated price paths, bubble components, and crash statistics.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 12  # Monthly

    # Monthly parameters
    mu = mu_d * dt
    sigma = sigma_d * np.sqrt(dt)
    rf_m = r_f * dt
    discount = np.exp(-rf_m)

    results = {
        "prices": np.zeros((n_sims, n_periods)),
        "fundamentals": np.zeros((n_sims, n_periods)),
        "bubbles": np.zeros((n_sims, n_periods)),
        "max_drawdowns": np.zeros(n_sims),
    }

    for sim in range(n_sims):
        # Dividend process (GBM)
        D = np.zeros(n_periods)
        D[0] = 1.0
        shocks = rng.normal(0, 1, n_periods)

        for t in range(1, n_periods):
            D[t] = D[t - 1] * np.exp(mu - 0.5 * sigma ** 2 + sigma * shocks[t])

        # Fundamental value (Gordon growth)
        PDV = D / max(rf_m - mu, 0.001)

        # Static investor wealth (grows with dividends + drift)
        V_static = np.zeros(n_periods)
        V_static[0] = theta * PDV[0]
        for t in range(1, n_periods):
            V_static[t] = V_static[t - 1] * (1 + mu + sigma * shocks[t])

        # Equilibrium price: P = PDV + theta * V_static
        P = PDV + theta * V_static
        bubble = theta * V_static

        results["prices"][sim] = P
        results["fundamentals"][sim] = PDV
        results["bubbles"][sim] = bubble

        # Max drawdown
        cummax = np.maximum.accumulate(P)
        drawdowns = (P - cummax) / cummax
        results["max_drawdowns"][sim] = np.min(drawdowns)

    return results


def stress_scenarios(sim_results, scenarios=None):
    """
    Apply stress scenarios to simulation results.

    Parameters
    ----------
    sim_results : dict
        From structural_simulation.
    scenarios : list of dict
        Each dict specifies a stress scenario.

    Returns
    -------
    DataFrame comparing outcomes across scenarios.
    """
    if scenarios is None:
        scenarios = [
            {"name": "Baseline", "theta_shock": 0, "dividend_shock": 0},
            {"name": "Mass redemption (-20% static)", "theta_shock": -0.20, "dividend_shock": 0},
            {"name": "Credit crunch", "theta_shock": -0.10, "dividend_shock": -0.15},
            {"name": "VIX spike (vol doubles)", "theta_shock": 0, "dividend_shock": 0, "vol_mult": 2},
        ]

    results = []
    for scenario in scenarios:
        # Simple multiplier analysis
        theta_effect = 1 + scenario.get("theta_shock", 0)
        div_effect = 1 + scenario.get("dividend_shock", 0)

        # Approximate price impact
        bubble_avg = np.mean(sim_results["bubbles"][:, -1])
        fund_avg = np.mean(sim_results["fundamentals"][:, -1])

        new_price = fund_avg * div_effect + bubble_avg * theta_effect
        old_price = fund_avg + bubble_avg
        price_change = (new_price - old_price) / old_price

        results.append({
            "scenario": scenario["name"],
            "price_change": price_change,
            "fundamental_change": div_effect - 1,
            "bubble_change": theta_effect - 1,
            "amplification": price_change / (div_effect - 1) if div_effect != 1 else np.nan,
        })

    return pd.DataFrame(results)


def run_e2(aggregate_panel=None, V_t=None, daily_returns=None,
           save_outputs=True):
    """
    Run the full E2 experiment.

    Returns
    -------
    dict with all E2 results.
    """
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("EXPERIMENT E2: Stress-Testing the Rational Bubble")
    logger.info("=" * 60)

    results = {}

    # --- Load V_t ---
    if V_t is None:
        cached_vt = load_cache("A1_V_t")
        if cached_vt is not None:
            V_t = cached_vt["V_t"]

    # --- Crisis identification ---
    if aggregate_panel is not None and "vwretd" in aggregate_panel.columns:
        logger.info("Step 1: Identifying crisis events...")
        crises = identify_crisis_events(aggregate_panel["vwretd"], threshold=-0.05)
        results["crises"] = crises
        logger.info(f"  Found {len(crises)} crisis months")

        # Crisis bubble response
        if V_t is not None:
            logger.info("Step 2: Crisis bubble response...")
            response = crisis_bubble_response(V_t, crises["date"].values)
            results["crisis_response"] = response
            if response["n_events"] > 0:
                logger.info(f"  Max decline = {response['max_decline']:.4f}")

    # --- Structural simulation ---
    logger.info("Step 3: Structural simulation...")
    sim = structural_simulation()
    results["simulation"] = sim
    logger.info(f"  Mean max drawdown = {np.mean(sim['max_drawdowns']):.4f}")

    # --- Stress scenarios ---
    logger.info("Step 4: Stress scenarios...")
    stress = stress_scenarios(sim)
    results["stress_scenarios"] = stress
    logger.info(f"\n{stress.to_string()}")

    # --- Save ---
    if save_outputs:
        to_csv(stress, "e2_stress_scenarios.csv")
        if "crises" in results:
            to_csv(results["crises"], "e2_crisis_events.csv")

        # Figure: simulation paths
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sample paths
        ax = axes[0, 0]
        for i in range(min(20, sim["prices"].shape[0])):
            ax.plot(sim["prices"][i], alpha=0.3, color="#2C3E50", linewidth=0.5)
        ax.set_title("Simulated Price Paths (20 samples)")
        ax.set_xlabel("Month")

        # Bubble share
        ax = axes[0, 1]
        bubble_share = sim["bubbles"] / sim["prices"]
        ax.plot(np.mean(bubble_share, axis=0), color="#E74C3C", linewidth=2)
        ax.fill_between(range(sim["prices"].shape[1]),
                         np.percentile(bubble_share, 5, axis=0),
                         np.percentile(bubble_share, 95, axis=0),
                         alpha=0.2, color="#E74C3C")
        ax.set_title("Bubble Share of Price (theta*V / P)")
        ax.set_ylabel("Bubble / Price")

        # Drawdown distribution
        ax = axes[1, 0]
        ax.hist(sim["max_drawdowns"], bins=50, color="#2C3E50", alpha=0.7)
        ax.axvline(x=np.mean(sim["max_drawdowns"]), color="#E74C3C",
                    linestyle="--", label=f"Mean: {np.mean(sim['max_drawdowns']):.1%}")
        ax.set_title("Distribution of Max Drawdowns")
        ax.set_xlabel("Max Drawdown")
        ax.legend()

        # Stress scenarios bar chart
        ax = axes[1, 1]
        colors = ["#2ECC71", "#F39C12", "#E74C3C", "#9B59B6"]
        ax.bar(range(len(stress)), stress["price_change"] * 100, color=colors[:len(stress)])
        ax.set_xticks(range(len(stress)))
        ax.set_xticklabels(stress["scenario"], rotation=45, ha="right")
        ax.set_ylabel("Price Change (%)")
        ax.set_title("Stress Scenario Results")

        fig.tight_layout()
        save_fig(fig, "e2_stress_testing.pdf")

    logger.info("=" * 60)
    logger.info("E2 COMPLETE")
    logger.info("=" * 60)

    return results
