# Comprehensive Research Experiment Agenda

## Based on "Asset Pricing with Dynamic and Static Investors" (Jappelli 2025)

---

## Executive Summary

This document outlines a comprehensive research agenda of 12 focused experiments designed to validate, extend, and stress-test the theoretical and empirical findings of Jappelli (2025). The experiments are organized into five thematic blocks:

- **Block A: Aggregate Price Pressure** — Direct tests of Proposition 1 and the rational bubble mechanism
- **Block B: Cross-Sectional Effects** — Tests of stock-level price pressure heterogeneity (Proposition 2)
- **Block C: Time-Series Dynamics** — Investigation of bad-times amplification and flow-price relationships
- **Block D: Macro-Elasticity Decomposition** — Decomposing static vs. dynamic flow impact on prices
- **Block E: Extensions and Stress Tests** — Challenging assumptions and exploring new markets

The agenda leverages proprietary datasets (Haddad et al. 2025 demand system estimates; Jiang et al. 2025 index data and flow IVs) as key research advantages. Timeline ranges from 6 months (quick validation studies) to 24+ months (ambitious multi-year projects).

---

# BLOCK A: AGGREGATE PRICE PRESSURE

## Experiment A1: Direct Test of the Rational Bubble Mechanism in θV_t

**Type:** Validate

**Hypothesis:** 
Static investor equity allocations (θ) create a rational, risk-adjusted martingale component θV_t in aggregate equity prices. This bubble (i) survives after controlling for fundamental dividends, (ii) satisfies the martingale property, and (iii) violates the transversality condition necessary for convergence to fundamentals.

**Gap Addressed:** 
Jappelli (2025) proves θV_t is a martingale in theory but provides limited direct empirical testing of this component's time-series properties and orthogonality to fundamentals. This experiment provides direct econometric evidence of the bubble's existence and properties.

**Method:**

Decompose aggregate equity value into three components:
$$P_t = \text{PDV}_t + \theta V_t + \varepsilon_t$$

Where:
- $\text{PDV}_t$ = present discounted value of dividends (estimated via Gordon growth model or VAR-based method)
- $\theta$ = average static investor equity allocation (measured from mutual fund data)
- $V_t$ = market value added by static investor presence (residual)

**Primary tests:**
1. **Martingale test** (Jappelli's approach): Regress $V_{t+1} - V_t$ on $V_t$ and other price-pressure instruments. H0: slope = 0, R² ≈ 0.
2. **Transversality violation test**: Estimate $\mathbb{E}_t[\sum_{s=0}^{\infty} \delta^s V_{t+s}]$ using long-horizon VAR; if finite and positive, transversality fails.
3. **Fundamental orthogonality test**: Regress $V_t$ on lagged dividend yield, consumption growth, and other macro fundamentals; test that coefficients approach zero.

**Data Requirements:**
- Aggregate equity prices (CRSP index): 1960–2024
- Aggregate dividend series (CRSP): 1960–2024
- Estimated $\theta$ from mutual fund data (Jappelli data + CRSP Mutual Fund Summary Files): 1980–2024
- Macro factors (Fama-French library): consumption growth, real rates, Sharpe ratio
- Long-term Treasury yields (FRED): for discount rate calculation

**Identification Strategy:**
- **Assumption 1:** Dividends are fundamental and exogenous to investor type composition. Test via Granger causality (does static fund growth Granger-cause dividends?).
- **Assumption 2:** $\theta$ is predetermined with respect to $V_t$ shocks. Instrument $\theta$ with lagged inflows into index funds (demand-side shock).
- **Sensitivity:** Use multiple dividend models (Gordon, VAR, long-horizon regressions) to ensure robustness of PDV estimates.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| $P_t$ | Log S&P 500 price (or CRSP value-weighted index) | CRSP |
| $D_t$ | Aggregate real dividends | CRSP |
| $\text{PDV}_t$ | Estimated fundamental value | Calculated |
| $\theta_t$ | Static investor equity share | CRSP Mutual Fund Summary |
| $V_t$ | Residual: $P_t - \text{PDV}_t$ | Calculated |
| Sharpe ratio | Realized excess return / volatility | Calculated from CRSP |

**Robustness Checks:**
1. **Alternative fundamental estimates:** Use vector autoregression (VAR) with [dividend, price, consumption] to back out implied PDV; compare to Gordon-model PDV.
2. **Alternative static fund definitions:** Repeat using SD(equity allocation) ≤ 3% and ≤ 7% (vs. 5%) to ensure classification is not driving results.
3. **Subsample stability:** Re-run tests on pre-2008, 2008–2015, post-2015 subperiods to test stability across market regimes.

**Statistical Power:**
- Sample: N ≈ 240 months (1980–2024)
- Martingale test: Minimum detectable effect (MDE) for slope coefficient ≈ 0.10 at 80% power
- Transversality test: Requires long-horizon forecasts; MDE ≈ 1% of current market value
- Power limited by structural breaks (e.g., rise of index investing); mitigate via subsample analysis

**Timeline:** 6 months

**Expected Contribution:**
Provides direct empirical evidence that static investor allocations create a rational, non-fundamental price component consistent with no-arbitrage. Distinguishes the Jappelli bubble from irrational bubbles or sentiment-driven mispricing. Strengthens the theoretical foundation by showing the mechanism is empirically detectable and persistent.

**Risk Assessment:**
- **Dividend measurement error:** Dividend estimates depend on accounting choices and reinvestment assumptions. Mitigate by using multiple sources (CRSP, Compustat) and imputation models.
- **PDV estimation bias:** Long-horizon dividend forecasts are uncertain. Use multiple methods and report range of PDV estimates.
- **Structural breaks in θ:** The rise of passive investing post-2010 may cause regime shifts in θ dynamics. Segment analysis by decade.

---

## Experiment A2: Aggregate Price Pressure as a Function of Static Fund Inflows

**Type:** Validate + Extend

**Hypothesis:** 
Unexpected inflows into static funds (balanced mutual funds, target-date funds) increase aggregate equity prices beyond fundamentals, with magnitude proportional to the flow as a share of market capitalization. The price impact is temporary (reverts over months) but statistically significant, supporting the no-arbitrage-compatible bubble mechanism.

**Gap Addressed:** 
Jappelli demonstrates theoretically that θ creates price pressure, but does not quantify the elasticity of prices to changes in θ (i.e., flow-to-price transmission). This experiment provides a reduced-form estimate of aggregate price impact per unit of unexpected inflow.

**Method:**

**Reduced-form regression:**
$$\Delta P_t = \alpha + \beta_1 \cdot \frac{\Delta F^{\text{static}}_t}{MV_t} + \beta_2 \cdot \Delta y_t^{\text{fund}} + \gamma \mathbf{Z}_t + \varepsilon_t$$

Where:
- $\Delta P_t$ = log price change of aggregate market
- $\frac{\Delta F^{\text{static}}_t}{MV_t}$ = net inflows to static funds, scaled by market cap
- $\Delta y_t^{\text{fund}}$ = change in yield-to-maturity of bond allocation (mechanical return adjustment)
- $\mathbf{Z}_t$ = controls (lagged returns, VIX, Fed policy changes, earnings surprises)
- $\beta_1$ = price elasticity to static fund flows (primary coefficient of interest)

**Instrument $\frac{\Delta F^{\text{static}}_t}{MV_t}$** using:
- Lagged S&P 500/Nasdaq rebalancing dates (index reconstitution events, exogenous to prices)
- 401(k) contribution shocks (e.g., January concentration, year-end contributions)
- Regulatory changes (Volcker Rule implementation, fiduciary rule timing)

**Event-study variant:** Around index inclusion/exclusion events (Jiang et al. 2025 data), estimate abnormal price change for firms with large increase/decrease in static fund ownership.

**Data Requirements:**
- CRSP Mutual Fund Summary Files: monthly TNA and flows, 2000–2024
- Fund asset allocation time series: equity %, bond %, cash % (from Jappelli or CRSP Classification)
- Market cap and returns (CRSP index): daily and monthly, 2000–2024
- Regulatory event dates: Volcker Rule (2013), Fiduciary Rule (2015–2017), Dodd-Frank timeline
- 401(k) calendar (IRS rules) and payroll timing
- S&P 500/600 index reconstitution dates (Jiang et al. 2025 data if available)

**Identification Strategy:**
- **Validity of instruments:** Test relevance (first-stage F-stat > 10) and overidentification (Hansen J-test, p-value > 0.05).
- **Exclusion restriction:** Argue that reconstitution dates and 401(k) contribution timing are plausibly exogenous to contemporaneous price shocks, conditional on controls.
- **Confounding:** Control for aggregate dividend surprises, Fed announcements, and macro news to rule out that inflows are simply responding to good news.
- **Dynamics:** Use Newey-West standard errors to account for autocorrelation in residuals.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| $\Delta P_t$ | Month-over-month log price change, value-weighted index | CRSP |
| $\Delta F^{\text{static}}_t$ | Net new flows to static funds (TNA change - returns) | CRSP MF Summary |
| $MV_t$ | Total market cap of U.S. equities | CRSP |
| Equity allocation % | Fund's equity / total assets | CRSP Classification or Jappelli |
| VIX | Realized volatility index | CBOE/WRDS |
| Sharpe ratio | Monthly excess return / volatility | Calculated |

**Robustness Checks:**
1. **Alternative flow definitions:** Use only active flows (exclude reinvestment), use percentage flows (flow / prior-year TNA), use dollar flows (in $B).
2. **Lag structure:** Include lagged flows (1–3 months) to test for delayed response. Test whether β₁ is largest at lag 0 or lag 1.
3. **Subsample by regime:** Repeat analysis in pre-crisis (2000–2007), crisis (2008–2009), recovery (2010–2024) to test stability.
4. **Alternative instrument:** Use unexpected inflows as IV (residuals from predictive model of flows based on past returns, volatility); verify results hold.

**Statistical Power:**
- Sample: N ≈ 280 months (2000–2024)
- Typical inflow as % of market cap: 0.005–0.02% per month
- Target MDE for β₁: 0.50 bps per 0.01% flow (50 bps per 1% inflow)
- Power: 80% at 5% significance with this MDE

**Timeline:** 8 months

**Expected Contribution:**
Quantifies the magnitude of aggregate price impact from static investor inflows, bridging theory (Jappelli) and practice (flow management). Provides evidence that passive investing mechanically moves prices, with implications for market efficiency, passive portfolio construction, and counterparty risk (i.e., who is absorbing dynamic investors' rebalancing trades?).

**Risk Assessment:**
- **Reverse causality:** High prices may attract inflows (performance-chasing), biasing β₁ upward. Mitigate with IV approach and lagged instruments.
- **Omitted variables:** Macro news and earnings surprises may drive both prices and flows. Control for major macro releases and earnings surprise indices.
- **Measurement error in θ:** Fund equity allocations are reported quarterly or annually; interpolate conservatively and test robustness to interpolation method.
- **Sample period bias:** Passive investing was small pre-2000; analysis limited to 2000+ may not generalize to earlier periods.

---

# BLOCK B: CROSS-SECTIONAL EFFECTS

## Experiment B1: Static Ownership as a Determinant of Cross-Sectional Stock Returns

**Type:** Validate + Extend

**Hypothesis:** 
Stocks with higher static investor ownership have higher risk-adjusted valuations and lower expected returns (Proposition 2). Controlling for firm fundamentals, market cap, and other known return drivers, a 1 SD increase in static ownership is associated with a 1–3% annualized reduction in subsequent returns. This effect is stronger for stocks with low liquidity or high idiosyncratic volatility.

**Gap Addressed:** 
Jappelli (2025) derives cross-sectional predictions (Proposition 2) but provides limited out-of-sample empirical validation on individual stock returns. This experiment tests whether the predicted cross-sectional pricing effects materialize in actual portfolios, controlling for multiple confounds.

**Method:**

**Panel regression (Fama-MacBeth):**
$$r_{it} = \alpha_t + \beta_1 \cdot \text{Static Ownership}_{it-1} + \beta_2 \cdot \log(MV_{it-1}) + \beta_3 \cdot BM_{it-1} + \beta_4 \cdot \text{Mom}_{it-1} + \gamma \mathbf{Z}_{it-1} + \varepsilon_{it}$$

Where:
- $r_{it}$ = excess return (buy-and-hold) for stock i in period t
- $\text{Static Ownership}_{it-1}$ = lagged share of stock i held by static investors (%)
- Control variables $\mathbf{Z}_{it-1}$: idiosyncratic volatility, turnover, analyst coverage, short interest
- Run cross-sectional regression each month; report time-series mean and t-stat of slope coefficients (Fama-MacBeth method)

**Portfolio sorts variant:** Form quintile portfolios based on lagged static ownership; compute long-short (high − low ownership) equal-weighted and value-weighted excess returns, adjusted for Fama-French 5-factor model.

**Instrumental variables (if reverse causality suspected):**
- Instrument static ownership with: (i) S&P 500/600 index inclusion (Jiang et al. 2025), (ii) stock entry into Russell 1000 (quarterly reconstitution), (iii) lagged fund inflow shocks (from Experiment A2)
- Validity: these events increase static fund demand for the stock exogenously

**Data Requirements:**
- CRSP daily/monthly stock prices and returns: 1980–2024
- Compustat fundamentals (book value, earnings, cash flow): annual, 1980–2024
- CRSP Mutual Fund holdings (if available) or reconstructed from Jappelli/Haddad data: static ownership by stock, quarterly, 2004–2024
- Market cap, book-to-market, momentum, idiosyncratic volatility: calculated or from Fama-French library
- Analyst coverage, short interest, turnover: from IBES and CRSP
- S&P index rebalancing dates (Jiang et al. 2025)

**Identification Strategy:**
- **Exogeneity of ownership:** Assume that index inclusion events change static fund demand exogenously (exclusion restriction: index inclusion affects returns only through the demand channel, not through information).
  - **Test:** Show that index inclusion has no effect on future earnings surprises or earnings revisions (no information contamination).
  - **Sensitivity:** Estimate reduced-form (inclusion → ownership) and second-stage (ownership → returns) separately to isolate channels.
- **Confounding by firm characteristics:** Use Lasso or double/debiased ML to select control set from large set of firm characteristics (size, growth, profitability, investment); ensure results robust to control selection.
- **Cross-sectional dependence:** Cluster standard errors by firm and by month to account for both within-firm and within-month correlation.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| $r_{it}$ | Monthly or annual excess return (log, annualized) | CRSP |
| Static Ownership % | Stock i's ownership share among static funds | CRSP MF Holdings or Jappelli |
| Market Cap | Log market value of equity | CRSP |
| Book-to-Market | Book value / market cap | Compustat + CRSP |
| Momentum | Past 12-month return (excluding recent month) | CRSP |
| Idiosyncratic Vol | Residual volatility from Fama-French model | Calculated |
| Analyst Coverage | Number of analysts covering the stock | IBES |
| Short Interest | Shares short / shares outstanding | CRSP |

**Robustness Checks:**
1. **Control set variation:** (i) Exclude all Fama-French controls, (ii) include 3-factor only, (iii) include 5-factor, (iv) add sentiment or liquidity variables. Verify β₁ stable.
2. **Sample splits:** Repeat for large-cap (top 500), mid-cap (500–1500), and small-cap (bottom 1000) separately. Verify effect is present and consistent.
3. **Subperiod analysis:** 1980–1999 (pre-indexing boom), 2000–2007 (index rise), 2008–2015 (crisis & recovery), 2016–2024 (current). Verify effect strengthens over time as passive investing grows.
4. **Alternative ownership definitions:** Use (i) share of public float, (ii) intensive margin only (conditional on ownership), (iii) change in ownership ∆Ownership rather than level.
5. **Alternate outcome variables:** Use analyst-adjusted (expected returns from survey), implied returns (from dividend discount models), or risk-adjusted returns (4-factor or 5-factor alphas).

**Statistical Power:**
- Sample: N ≈ 2,000–3,000 stocks/month × 20 years = 480,000–720,000 stock-months
- Typical static ownership: 2–10% of public float
- Target MDE for β₁: −0.10 to −0.15 annualized return per 1 SD ownership (SD ≈ 3–5%)
- Power: 95%+ with this sample size and MDE

**Timeline:** 10 months

**Expected Contribution:**
Provides comprehensive evidence that the cross-sectional pricing predictions from Jappelli (2025) hold in real data, validating Proposition 2. Extends the theory by quantifying magnitude of effect, testing heterogeneity by firm characteristics, and identifying which stocks are most affected. Implications for: (i) passive investor performance, (ii) active manager alpha sources, (iii) rebalancing strategies.

**Risk Assessment:**
- **Measurement error in ownership:** Fund holdings reported quarterly; interpolate carefully. Verify results robust to holding assumption (linear interpolation vs. constant holding).
- **Selection bias in data availability:** Funds with larger AUM may have better-reported holdings; creates selection bias. Mitigate by weighting regressions by fund size or restricting to top funds.
- **Index inclusion endogeneity:** S&P 500 inclusion may be correlated with unobserved firm quality. Control for lagged profitability, growth, and valuation metrics.
- **Return measurement:** Dividends may not be promptly reinvested; verify using total return (ex-date adjusted) from CRSP.

---

## Experiment B2: Heterogeneous Static Ownership Effects by Firm Characteristics

**Type:** Extend

**Hypothesis:** 
The price impact of static ownership (Proposition 2 coefficient on $\theta q^*_{it}$) varies systematically with firm characteristics. Specifically:
- **High-beta stocks:** Static ownership has larger valuation effect (higher weight in index-following strategies means more demand)
- **Low-liquidity stocks:** Static ownership has larger price impact (harder for dynamic investors to arbitrage)
- **Low-idiosyncratic-volatility stocks:** Larger valuation effect (appeal to static investors seeking diversification)
- **High-index-weight stocks:** Larger effect (more overlap between static-investor and index-fund portfolios)

**Gap Addressed:** 
Jappelli (2025) notes heterogeneity in Proposition 2 but does not systematically decompose the cross-sectional effect by characteristics. This experiment tests whether the bubble mechanism has predictable heterogeneity aligned with economic intuition and data.

**Method:**

**Conditional effect estimation via interaction models:**
$$r_{it} = \alpha_t + \beta_1 \cdot SO_{it-1} + \beta_2 \cdot SO_{it-1} \times \text{High-Beta}_{it-1} + \beta_3 \cdot SO_{it-1} \times \text{Low-Liq}_{it-1} + \gamma \mathbf{Z}_{it-1} + \varepsilon_{it}$$

Where:
- $SO_{it-1}$ = static ownership (centered for interpretation)
- Interaction terms capture differential effects
- High-Beta, Low-Liq, etc. = indicator or z-score of characteristic

**Double-sorting approach:** 
1. Sort stocks by characteristic (e.g., quartiles of beta)
2. Within each characteristic quartile, re-sort by static ownership quintile
3. Compute long-short (high SO − low SO) returns within each characteristic group
4. Test whether long-short spread varies significantly across characteristic quartiles

**Quantile regression variant (if effect is heterogeneous across return distribution):**
$$Q_\tau(r_{it}) = \alpha_t + \beta_1(\tau) \cdot SO_{it-1} + \text{controls} + \varepsilon_{it}$$

Estimate β₁(τ) for τ = {0.25, 0.50, 0.75} to test whether static ownership effect is larger during market stress (lower quantiles) or booms (upper quantiles).

**Data Requirements:**
- All data from Experiment B1, plus:
- Beta: estimated from rolling 60-month regression of stock return on market return (CRSP)
- Liquidity measures: bid-ask spread (if available from CRSP), or Amihud illiquidity measure (|return| / volume), Turnover (volume / shares outstanding)
- Idiosyncratic volatility: residual variance from Fama-French 3-factor model
- Index weight: stock market cap / total market cap (S&P 500 or Russell 2000 index weight)

**Identification Strategy:**
- **Conditional exogeneity:** Assume that, conditional on observable characteristics, static ownership is exogenous (or can be instrumented as in Experiment B1).
- **Treatment effect heterogeneity:** Use generalized random forest (GRF) or causal forest estimators to flexibly estimate heterogeneous treatment effects across characteristics (allows non-parametric interactions).
- **Falsification test:** Test that ownership effects on returns are stronger for stocks with high index weight (predicted), but ownership has no effect on dividend per-share announcements (should be fundamental, unaffected by ownership).

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| Static Ownership | As in Experiment B1 | CRSP MF Holdings or Jappelli |
| Beta | 60-month rolling covariance with market / market variance | CRSP |
| Amihud Illiquidity | $\frac{1}{D} \sum_{d=1}^D \frac{\|r_{id}\|}{V_{id}}$ (D=days, V=volume) | CRSP |
| Turnover | Monthly share volume / shares outstanding | CRSP |
| Idiosyncratic Volatility | Std dev of residuals from FF3 model | Calculated |
| Index Weight | Market cap of stock / total index market cap | Calculated |
| Analyst Dispersion | Std dev of analyst earnings forecasts / mean forecast | IBES |

**Robustness Checks:**
1. **Placebo tests:** Verify that static ownership does NOT have differential effects on non-price outcomes (e.g., analyst earnings revisions, insider trading volume). Should be null.
2. **Alternative characteristic definitions:** Use quintiles vs. terciles, continuous vs. binary (high/low split at median).
3. **Alternative methods:** Repeat double-sort using equal-weighted and value-weighted returns; verify direction consistent.
4. **Subsample by ownership level:** Repeat analysis separately for stocks with low (0–2%), medium (2–5%), and high (>5%) static ownership. Test linearity vs. non-linearity of effect.
5. **Exclude index overlap:** Drop S&P 500 constituents and test whether effect remains for Russell 2000. If effect is purely mechanical (index arbitrage), should be much weaker.

**Statistical Power:**
- Sample: same as Experiment B1 (≈500k stock-months)
- Interaction effect MDE: 0.05 to 0.10 annualized return per interaction (smaller than main effect, but adequately powered at this sample size)
- Power: 80%+ for main interactions

**Timeline:** 8 months

**Expected Contribution:**
Decomposes the cross-sectional bubble mechanism into economically meaningful components, showing which types of stocks are most susceptible to static investor demand. Provides practical guidance for: (i) portfolio managers (identifying alpha from mispricing), (ii) index providers (understanding demand for index constituents), (iii) policymakers (systemic risk from passive investing concentration).

**Risk Assessment:**
- **Multicollinearity:** Ownership and firm characteristics (e.g., size, beta) may be correlated. Verify VIF < 5 for key variables.
- **Multiple testing:** With multiple characteristic interactions, risk of false discovery. Use multiple testing correction (Benjamini-Hochberg FDR) or pre-register hypotheses.
- **Functional form:** Interaction effects are linear approximations; true effect may be non-linear. Robustness-check with spline or polynomial terms.

---

# BLOCK C: TIME-SERIES DYNAMICS

## Experiment C1: Bad-Times Amplification of Static Fund Price Pressure

**Type:** Validate + Extend

**Hypothesis:** 
The asset classification effect (price pressure from static fund ownership) is counter-cyclical: it amplifies during market downturns when the Sharpe ratio is low. Quantitatively, a 1 SD decrease in Sharpe ratio amplifies the static ownership effect on prices by 40–80% (i.e., low Sharpe ratio periods see 2–3× larger bubble component). This reflects diminished willingness of dynamic investors to arbitrage away the static fund bubble when returns are poor and volatility is high.

**Gap Addressed:** 
Jappelli (2025) documents correlation of −0.65 (first differences) between static ownership share and Sharpe ratio, suggesting time-varying bubble magnitude. However, the mechanism (why does amplification occur?) and the quantitative dynamic relationship are underexplored. This experiment provides direct evidence of counter-cyclical bubble dynamics.

**Method:**

**Time-varying parameter model (TVP-VAR or state-space):**
$$V_t = \alpha_t(\text{Sharpe}_t) + \beta_t(\text{Sharpe}_t) \cdot \theta_t + \gamma_t(\text{Sharpe}_t) \cdot Z_t + \varepsilon_t$$

Where:
- $V_t$ = residual (non-fundamental) component of prices (estimated from Experiment A1)
- $\alpha_t(\text{Sharpe}_t)$ = intercept that varies with Sharpe ratio
- $\beta_t(\text{Sharpe}_t)$ = coefficient on static ownership, which varies with Sharpe ratio
- $Z_t$ = controls (risk premium, credit spreads, VIX)

**Implementation:**
1. Use local-linear estimation or moving-window regression (36-month rolling window) to estimate time-varying β
2. Regress time-varying β on lagged Sharpe ratio and other macro variables
3. Primary coefficient of interest: $\partial \beta_t / \partial \text{Sharpe}_{t-1}$ (elasticity of bubble effect to Sharpe ratio)

**Alternative: Quantile regression on Sharpe ratio bins:**
$$V_t = \alpha + \beta(\text{Sharpe}_t \text{ quartile}) \cdot \theta_t + \varepsilon_t$$

Estimate separate coefficients for each Sharpe ratio quartile (lowest = market stress, highest = calm markets). Test H0: β coefficients are equal (no amplification).

**Bayesian Local Level model (structural time series):**
$$V_t = \mu_t + \varepsilon_t, \quad \mu_t = \mu_{t-1} + \nu_t$$
$$\nu_t \sim N(0, \sigma^2_\nu(Sharpe_t))$$

Where level variance $\sigma^2_\nu$ scales with Sharpe ratio (bad times → higher variance → larger deviations). Use Kalman filter to estimate time-varying level and test whether level volatility co-varies with Sharpe ratio.

**Data Requirements:**
- $V_t$ (residual bubble component) from Experiment A1: monthly, 1980–2024
- Sharpe ratio: monthly realized (return − risk-free rate) / volatility, 1980–2024
- Static ownership $\theta_t$: monthly average (interpolated from quarterly mutual fund data), 2000–2024 (or extend back if available)
- Control variables: credit spread (BAA − AAA), VIX, term spread (10y − 2y Treasury), market beta risk premium
- Fed policy rate and expectations (from FRED)

**Identification Strategy:**
- **Causality direction:** Sharpe ratio and static ownership are both endogenous. Use Granger causality tests to establish lead-lag relationships; verify that lagged Sharpe ratio predicts future β (not vice versa).
- **Exogeneity of Sharpe ratio:** Assume Sharpe ratio is predetermined with respect to contemporaneous shocks to V (i.e., $\text{Sharpe}_{t-1}$ is exogenous in regression of V_t). Sensitivity: use lagged Sharpe ratio exclusively in main regression.
- **Structural breaks:** Test for regime switches using Chow test or Markov regime-switching model. Verify results robust to structural break dates.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| $V_t$ | Residual (non-fundamental) value | Experiment A1 |
| Sharpe Ratio | Realized (r − rf) / σ, monthly | CRSP, FRED |
| θ_t | Static ownership, monthly | CRSP MF Summary |
| VIX | 30-day implied volatility (S&P 500) | CBOE |
| Credit Spread | BAA − AAA bond yield spread | FRED |
| Term Spread | 10-year − 2-year Treasury yield | FRED |
| Fed Funds Rate | Effective federal funds rate | FRED |

**Robustness Checks:**
1. **Alternative Sharpe measures:** Use (i) forward-looking implied Sharpe (from option prices), (ii) GARCH-estimated volatility (instead of realized), (iii) 12-month rolling Sharpe.
2. **Alternative market conditions indicator:** Replace Sharpe ratio with (i) NBER recession indicator, (ii) credit spread, (iii) VIX level. Verify effects are robust to market stress measure.
3. **Lag structure:** Test whether contemporaneous Sharpe affects β (mechanical relationship) vs. lagged Sharpe (causal), and whether effect decays over time.
4. **Sample splits:** Repeat analysis in normal-times (Sharpe > 0.5) vs. stress-times (Sharpe < 0.2) as discrete subsamples.
5. **Falsification test:** Verify that Sharpe ratio does NOT significantly affect the relationship between (i) fundamental dividend component and firm profitability (should be time-invariant), or (ii) other unrelated price movements (e.g., bond prices).

**Statistical Power:**
- Sample: N ≈ 240 months (1980–2024)
- Sharpe ratio time-variation: SD ≈ 0.30–0.50 annualized
- Target MDE for amplification effect: 40% increase in β per 1 SD Sharpe drop
- Power: 75–80% for detecting this effect at N=240 and 5% significance

**Timeline:** 9 months

**Expected Contribution:**
Provides dynamic characterization of the bubble mechanism, showing it is not constant but amplifies during market stress. Supports the economic intuition that arbitrage is limited in bad times. Has implications for: (i) procyclicality of price pressures (destabilizing), (ii) passive investor systemic risk (larger impact when markets are stressed), (iii) rebalancing strategies for static funds (should they be less rigid in crises?).

**Risk Assessment:**
- **Lookback bias:** If time-varying β is estimated using rolling windows that include future information, results may be biased. Mitigate by using expanding windows or Kalman filter (real-time estimates).
- **Spurious correlation:** Sharpe ratio and V may both be low during crises (mechanical relationship) without one causing the other. Sensitivity-test using lagged Sharpe only.
- **Data quality:** Monthly aggregates smooth out high-frequency dynamics; alternative with weekly or daily data if available.

---

## Experiment C2: Inflow-Return Causality and Bubble Dynamics

**Type:** Extend

**Hypothesis:** 
Inflows into static funds causally increase equity prices (supporting the rational bubble mechanism), but the effect decays over time. A $100M unexpected inflow into static funds today increases market prices by 5–15 bps immediately, but price reverts by 50% within 6 months as dynamic investors gradually rebalance. The half-life of price impact is 6–12 months, shorter in high-Sharpe-ratio periods (when arbitrage is strong).

**Gap Addressed:** 
Experiment A2 tests static flow impact on aggregate prices but uses reduced-form OLS/IV. This experiment uses vector autoregression (VAR) and impulse response functions (IRFs) to characterize dynamic price adjustment and test whether it is consistent with the rational bubble model (i.e., slow mean-reversion due to martingale property) or inconsistent (e.g., quick mean-reversion suggesting mechanical mispricing that corrects).

**Method:**

**Structural VAR with Cholesky identification:**
$$\mathbf{X}_t = \mathbf{A}_1 \mathbf{X}_{t-1} + \mathbf{A}_2 \mathbf{X}_{t-2} + \cdots + \mathbf{A}_p \mathbf{X}_{t-p} + \mathbf{B} \varepsilon_t$$

Where $\mathbf{X}_t = [\log P_t, \log F^{\text{static}}_t, Sharpe_t, VIX_t]'$ and $\varepsilon_t \sim N(0, I)$ are structural shocks.

**Ordering (Cholesky):** 
1. Sharpe ratio (most exogenous, macroeconomic)
2. VIX (volatility, exogenous to individual fund flows)
3. Static fund flows (respond to Sharpe ratio, VIX, but not contemporaneous prices)
4. Prices (respond to all)

Justification: flows are driven by macro conditions and investor sentiment, not instantaneous price feedback (flows settle over days/weeks).

**Impulse response:** Shock static flows by 1 SD ($\approx$ 0.5–1% of market cap), trace price response over 24 months. Compute:
- **Impact elasticity:** Price response (%) to 1% inflow in month 0
- **Half-life:** Months until price impact decays to 50% of initial
- **Long-run effect:** Cumulative price response (test if → 0 as rational bubble predicts, or → permanent value, as mechanical mispricing might suggest)

**Time-varying VAR:** Estimate VAR separately in high-Sharpe (calm) and low-Sharpe (stress) regimes. Compare IRFs across regimes. Hypothesis: half-life shorter in high-Sharpe regime (stronger arbitrage).

**Alternative: Local Projections (LP) approach** (more robust to lag-length misspecification):
$$r_{t+h} = \alpha_h + \beta_h(h) \cdot \Delta F^{\text{static}}_t / MV_t + \gamma_h \mathbf{Z}_{t-1} + \varepsilon_{t+h}$$

For h = 0, 1, 2, ..., 24 (months ahead). Coefficient $\beta_h(h)$ gives cumulative return response h months after shock. Plot $\beta_h(h)$ to visualize impulse response.

**Data Requirements:**
- Monthly returns, static fund flows, Sharpe ratio, VIX: 2000–2024 (minimum 20 years for VAR stability)
- Market cap and aggregate dividend yields (for controlling for dividend strips)
- Federal funds rate, credit spreads (control variables)
- Detailed mutual fund inflow data (CRSP Mutual Fund Summary or reconstructed from net flows)

**Identification Strategy:**
- **Assumption 1 (Recursivity):** Flows do not respond within-month to prices. Test: regress monthly flows on current-month return; should be insignificant when flow is measured as end-of-month position change.
- **Assumption 2 (Exogeneity of flows):** Static fund flows driven by external demand shocks (performance-chasing, contributions), not strategic arbitrage. Test by verifying flows Granger-cause prices, not vice versa.
- **Robustness to ordering:** Re-estimate with alternative Cholesky orderings (e.g., flows before Sharpe, prices before flows); verify qualitative results stable.
- **Robustness to lag length:** Use Akaike and Schwarz criteria to select optimal p; re-estimate with p ± 2; verify IRF shapes stable.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| $r_t$ | Monthly excess return (log) | CRSP |
| $\Delta F^{\text{static}}_t$ | Net inflows to static funds | CRSP MF Summary |
| Sharpe_t | Realized Sharpe ratio, monthly | Calculated |
| VIX_t | Implied volatility index | CBOE/WRDS |
| Fed Rate | Effective federal funds rate | FRED |
| Credit Spread | BAA − AAA | FRED |

**Robustness Checks:**
1. **Alternative flow definitions:** Net vs. gross flows, dollar flows vs. percentage flows, static funds only vs. static + quasi-passive (low turnover).
2. **Sub-period IRFs:** Estimate separate VAR for pre-2008, 2008–2015, post-2015; verify IRF shapes and half-lives stable.
3. **Exogeneity test:** Check whether flows respond to lagged prices (would indicate performance-chasing and reverse causality). If significant, use lags-only specification.
4. **Falsification:** Estimate IRF of flows to price shocks (should be small or wrong sign if flows are exogenous); verify.
5. **Sign restrictions:** Instead of Cholesky, use sign restrictions (flows increase on Sharpe shocks, etc.); verify IRF robust to identification method.

**Statistical Power:**
- Sample: N ≈ 280 months (2000–2024)
- Inflow variability: SD(ΔF/MV) ≈ 0.01–0.02% per month
- Target MDE for initial price response: 0.05–0.10% (5–10 bps) per 0.01% inflow
- Power: 85%+ (VAR with N=280 and moderate effect size)

**Timeline:** 10 months

**Expected Contribution:**
Characterizes the dynamic adjustment of prices to static fund flows, testing whether the mechanism is consistent with a rational martingale bubble (slow mean-reversion, no long-run impact) or inconsistent (fast reversion, suggesting mispricing). Provides investors with guidance on timing rebalancing around large flows. Documents regime-dependent arbitrage strength, with implications for market microstructure and passive investor systemic risk.

**Risk Assessment:**
- **Non-stationarity:** Price levels may be I(1); verify via ADF test. Use log-differences in VAR if needed.
- **Model specification:** VAR assumes linear relationships; true dynamics may be non-linear (especially in stress regimes). Robustness-check with regime-switching VAR or threshold VAR.
- **Lag length:** Small sample (280 obs) limits lag length; high p (>4) may overfit. Use information criteria and check residual autocorrelation.
- **Structural breaks:** 2008 crisis and 2020 COVID shock are massive outliers. Robustness-check excluding these periods; also estimate using robust regression (downweight outliers).

---

# BLOCK D: MACRO-ELASTICITY DECOMPOSITION

## Experiment D1: Differential Price Elasticity of Static vs. Dynamic Flows

**Type:** Validate + Extend

**Hypothesis:** 
Static investor flows have a higher price impact per dollar than dynamic investor flows. Specifically, a $1B exogenous inflow to static funds increases prices by 3–5 bps, while a $1B flow to active/dynamic funds increases prices by 0.5–1 bp (6–10× smaller). This difference arises because static flows are committed demand (sticky allocations), while dynamic flows are endogenously determined by portfolio optimization and face binding liquidity constraints. The elasticity difference implies that market impact is not uniform and depends on flow composition.

**Gap Addressed:** 
Jappelli's theoretical model predicts differential impact of static vs. dynamic flows, but empirical literature typically treats all flows equivalently. This experiment quantifies the elasticity differential and tests whether the decomposition explains unexplained heterogeneity in past flow-impact studies (e.g., why equity mutual fund flows have lower impact than predicted).

**Method:**

**Separate regressions for static vs. dynamic flows:**
$$\Delta P_t = \alpha + \beta^{\text{static}} \cdot \frac{\Delta F^{\text{static}}_t}{MV_t} + \beta^{\text{dynamic}} \cdot \frac{\Delta F^{\text{dynamic}}_t}{MV_t} + \gamma \mathbf{Z}_t + \varepsilon_t$$

Where:
- $\Delta F^{\text{static}}_t$ = net inflows to static funds (measured from CRSP Mutual Fund Summary, SD(equity allocation) ≤ 5%)
- $\Delta F^{\text{dynamic}}_t$ = net inflows to actively managed funds (high turnover, active share > 50%)
- Test H0: $\beta^{\text{static}} = \beta^{\text{dynamic}}$ (equality of elasticity)

**Instrument both flow terms** using:
- For static flows: S&P 500 index reconstitution events, 401(k) contribution timing, regulatory changes (as in Experiment A2)
- For dynamic flows: mutual fund family IPOs, mergers that expand fund product line, changes in expense ratios (competitive shocks), redemption requests from large institutional clients (if data available)
- Form matrix of instruments; test rank, validity, and overidentification

**Haddad et al. (2025) demand-system approach:** 
If demand system estimates (elasticity of fund flows to past returns, expense ratios, fund characteristics) are available, use these to decompose realized flows into "rational" flows (explained by fundamentals and past performance) vs. "exogenous" flows (residual). Regress price changes on exogenous flows separately for static vs. dynamic funds.

**Cross-sectional variant (stock level):**
$$r_{it} = \alpha + \beta^{\text{static}} \cdot \text{StaticFlowExposure}_{it-1} + \beta^{\text{dynamic}} \cdot \text{DynamicFlowExposure}_{it-1} + \varepsilon_{it}$$

Where StaticFlowExposure is the expected impact of aggregate static fund flows on stock i (based on fund holding patterns), and similarly for dynamic. Verify that static flow exposure has larger return impact.

**Data Requirements:**
- CRSP Mutual Fund Summary Files (monthly TNA, flows, asset allocation): 2000–2024
- Fund classification into active vs. passive (Active Share, turnover, strategy classification from CRSP)
- Mutual fund holdings (CRSP or SEC filings) to construct StaticFlowExposure by stock
- Market cap and returns (CRSP): monthly, 2000–2024
- Instrument dates: S&P rebalancing (Jiang et al. 2025), 401(k) calendar (IRS)
- **If available:** Haddad et al. (2025) demand-system parameter estimates, Jiang et al. (2025) index inclusion IVs
- Fund-level control variables: expense ratio, past returns, fund age, fund family size

**Identification Strategy:**
- **Distinguishing static vs. dynamic flows:** Define static funds as those with SD(equity allocation) ≤ 5% and relatively constant holdings (rebalance annually/semi-annually). Define dynamic as those with high turnover (>50% annual) or active share > 50%. Verify classification robust to cutoffs.
- **Validity of instruments:**
  - For static flows: S&P inclusion is exogenous to individual flows (exclusion restriction: inclusion affects prices only through static fund demand, not through information). Test by showing inclusion has no effect on firm fundamentals (earnings surprises) contemporaneously.
  - For dynamic flows: Merger/competition shocks are exogenous to market prices conditional on controls. Test by showing shocks have no effect on aggregate market conditions (Sharpe ratio, VIX).
- **Confounding:** Control for contemporaneous changes in fund expenses, performance, and aggregate market conditions (returns, volatility).
- **Dynamics:** Use multi-period lags to account for delayed fund flow responses and price adjustments.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| $\Delta P_t$ | Log price change, S&P 500 | CRSP |
| $\Delta F^{\text{static}}_t$ | Net flows to static-allocation funds | CRSP MF Summary |
| $\Delta F^{\text{dynamic}}_t$ | Net flows to active-management funds | CRSP MF Summary |
| StaticFlowExposure_{it} | Sum of static fund holdings × (aggregated flow shock) | Constructed |
| DynamicFlowExposure_{it} | Sum of dynamic fund holdings × (aggregated flow shock) | Constructed |
| Active Share | % of fund portfolio that differs from benchmark | CRSP |
| Turnover | Annual portfolio turnover ratio | CRSP |

**Robustness Checks:**
1. **Alternative classifications:** Use (i) explicit passive flag (if available), (ii) expense ratio quartiles (low expense = likely passive), (iii) index tracking error (low tracking error = passive).
2. **Lag structure:** Include lagged flows (1–3 months) to test whether price response is immediate or delayed, and whether it differs between static and dynamic.
3. **Subsample by flow size:** Repeat analysis separately for very large flows (top decile) vs. small flows, to test whether elasticity varies by flow magnitude (nonlinearities).
4. **Exclude confounding events:** Drop months with major Fed announcements, earnings surprises, or macro data releases; verify elasticity differences persist.
5. **Instrument strength:** Report first-stage F-stats for each instrument set; verify F > 10 for valid inference. Try alternative instrument sets.

**Statistical Power:**
- Sample: N ≈ 280 months (2000–2024)
- Typical flow composition: ~20% static, ~80% dynamic (by AUM)
- Static flow elasticity: β^static ≈ 3 bps per 0.1% inflow; dynamic ≈ 0.5 bps
- MDE for elasticity ratio: 3–5× difference is detectable at 80% power

**Timeline:** 11 months

**Expected Contribution:**
Provides empirical evidence that flow elasticity is not uniform but depends on investor type, directly validating Jappelli's prediction that static flows have outsized price impact. Extends the literature by decomposing historically observed flow-impact heterogeneity into a theoretically grounded framework. Implications for: (i) institutional investors' trading (static flows are "tougher" to trade against), (ii) market-making (different hedging strategies for static vs. dynamic flows), (iii) passive investor systemic risk (larger price impact per unit of passive flows).

**Risk Assessment:**
- **Classification error:** Distinguishing static vs. dynamic funds is imperfect (some "active" funds may be closet indexers; some "static" may be tactical). Sensitivity-test with alternative classification schemes.
- **Selection bias in fund-level data:** Firms with better reporting infrastructure may have superior holdings data. Weight analysis by fund size to address selection.
- **Reverse causality in fund flows:** Flows may respond to recent returns, biasing elasticity estimates. Instrument robustly and verify results with Granger causality tests.
- **Measurement of "exogenous" flows:** True exogenous flows are unobservable; even instrumented flows may be endogenous to unobserved demand shifters. Sensitivity-test robustness to different IV sets.

---

## Experiment D2: State-Dependent Elasticity and Spillovers to Other Assets

**Type:** Extend

**Hypothesis:** 
The price elasticity of static fund flows varies with market state and exhibits spillovers to other asset classes. Specifically:
- **Cross-asset spillovers:** A $1B inflow to equity index funds reduces bond prices (negative spillover) by 0.5–1.0 bps if the funds rebalance by selling bonds. Spillover magnitude is larger in stress periods (lower liquidity, larger rebalancing needs).
- **Spillovers within equities:** Index-heavy flows spill over to non-index stocks, as portfolio managers of non-indexed firms rebalance to maintain diversification or hedge factor exposures.
- **Regime dependence:** Elasticity is 2–3× larger in high-VIX / low-Sharpe regimes (stress times) compared to calm markets.

**Gap Addressed:** 
Jappelli focuses on equities; spillovers to bonds and other asset classes are unstudied. Additionally, the interaction between flow elasticity and market state is underexplored. This experiment extends the bubble mechanism to multi-asset settings and tests whether static flows create spillover risks across markets.

**Method:**

**Multivariate regression with cross-asset outcomes:**
$$\Delta P^{\text{bonds}}_t = \alpha + \beta^{eq \to bond} \cdot \frac{\Delta F^{\text{static,equity}}_t}{MV^{eq}_t} + \gamma^{\text{bond}} \mathbf{Z}_t + \varepsilon_t$$

Where ΔP^bonds is the price change of a broad bond index (Barclays Aggregate or equivalent). Coefficient $\beta^{eq \to bond}$ measures spillover from equity flows to bonds.

**Spillovers within equities:**
$$r^{\text{non-index}}_t = \alpha + \beta \cdot \frac{\Delta F^{\text{static}}_t}{MV^{eq}_t} + \gamma \text{Index Weight}_{it} + \text{controls} + \varepsilon_t$$

Test whether non-index stocks have negative returns when index-fund flows are strong (as portfolio rebalancing sells non-index stocks to maintain index underweight).

**State-dependent elasticity (triple interaction):**
$$\Delta P_t = \alpha + \beta_0 \cdot \frac{\Delta F^{\text{static}}_t}{MV_t} + \beta_1 \cdot \frac{\Delta F^{\text{static}}_t}{MV_t} \times \mathbb{1}(\text{high-VIX}_t) + \text{controls} + \varepsilon_t$$

Coefficient $\beta_1$ measures additional elasticity in high-VIX regimes.

**Alternative: Time-varying elasticity via quantile regression on VIX quartiles:**
$$\Delta P_t = \alpha(VIX_t \text{ quartile}) + \beta(VIX_t \text{ quartile}) \cdot \frac{\Delta F^{\text{static}}_t}{MV_t} + \varepsilon_t$$

Estimate separate slopes for each VIX quartile; test for trend (does β increase monotonically with VIX?).

**Haddad et al. demand-system integration:** 
If demand-system estimates include cross-asset substitution elasticities, use these to predict spillover magnitudes; compare predicted vs. observed spillovers.

**Data Requirements:**
- Equity flows, prices (from prior experiments)
- Bond index prices and yields: Bloomberg Barclays Aggregate Index or equivalent (monthly, 2000–2024)
- Bond fund flows (CRSP Mutual Fund Summary, filtered to bond funds)
- VIX, credit spreads (CBOE, FRED)
- Multi-asset allocation of target-date and balanced funds (rebalancing weights between equities and bonds)
- Equity index weights (to construct "index stock" vs. "non-index stock" returns)

**Identification Strategy:**
- **Exogeneity of equity flows to bond market shocks:** Assume equity static flows are driven by equity-market factors (equity Sharpe ratio, equity volatility) and macro factors (interest rates), not bond-market conditions. Test by verifying equity flows do not Granger-cause bond yields.
- **Causal mechanism for spillover:** Model rebalancing explicitly. E.g., if $1B equity inflow causes target-date funds to rebalance, estimate the implied bond outflow based on fund target allocations (e.g., 60/40), then test whether bond prices move by the predicted amount.
- **Alternative spillover sources:** Control for correlated shocks to equities and bonds (e.g., risk-off events that hurt both). Include term-spread changes, credit-spread changes as controls.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| $\Delta P^{\text{bonds}}_t$ | Monthly price change, bond index | Bloomberg Barclays or FRED |
| $\Delta P^{\text{non-index}}_t$ | Return of non-index stocks (Russell 2000, for example) | CRSP |
| $\Delta F^{\text{static,equity}}_t$ | Net inflows to static equity funds | CRSP MF Summary |
| $\Delta F^{\text{static,bond}}_t$ | Net inflows to static bond funds | CRSP MF Summary |
| VIX_t | Implied volatility index | CBOE |
| High-VIX indicator | 1 if VIX > 75th percentile, 0 otherwise | Calculated |
| Bond Yield | 10-year Treasury yield or Agg yield | FRED |
| Term Spread | 10-year − 2-year Treasury | FRED |

**Robustness Checks:**
1. **Alternative bond indices:** Use 10-year Treasury yields vs. corporate bond spreads vs. aggregate bond index; verify spillover direction consistent.
2. **Lagged effects:** Include lagged flows to test whether spillovers are contemporaneous or delayed.
3. **Subsample by regime:** Estimate spillovers separately in calm markets (VIX < 15), normal markets (VIX 15–20), stress markets (VIX > 20); verify spillover magnitude increases.
4. **Falsification test:** Estimate spillovers from bond flows to equity prices; should be weaker or zero if causality runs equity → bond (not reverse).
5. **Alternative spillover mechanism:** Test whether spillovers work through (i) direct rebalancing (model fund allocations), or (ii) indirect portfolio hedging (managers hedge equity exposure through bond futures), or both.

**Statistical Power:**
- Sample: N ≈ 280 months
- Spillover elasticity: 0.5–1.0 bps per 0.1% equity inflow; MDE ≈ 0.3 bps
- Power: 70–75% for detecting spillovers in calm markets, 85%+ in stress markets (larger effect)

**Timeline:** 12 months

**Expected Contribution:**
Extends the static-investor bubble mechanism to multi-asset settings, documenting systemic implications (spillovers reduce diversification benefits, increase portfolio risk). Demonstrates that the macroeconomic impact of passive investing is larger than equity-focused analyses suggest. Provides policymakers with evidence of interconnectedness and systemic risk. Implications for: (i) fund rebalancing policies, (ii) regulatory stress-testing (passive investor flows as a market risk factor), (iii) multi-asset portfolio construction.

**Risk Assessment:**
- **Correlated shocks:** Equities and bonds move together in risk-off events; difficult to isolate flow-driven spillovers from fundamental shocks. Mitigate by including explicit macro controls and using weekly/daily data for finer resolution.
- **Fund rebalancing endogeneity:** Funds may time rebalancing to current market conditions; requires careful timing analysis and lagged analysis.
- **Non-linear spillovers:** Spillovers may be much larger during crisis (e.g., March 2020) than normal times; quantile regression accounts for this, but sample size in tail is small.

---

# BLOCK E: EXTENSIONS AND STRESS TESTS

## Experiment E1: International Validation and Market-Specific Heterogeneity

**Type:** Extend

**Hypothesis:** 
The Jappelli mechanism (static investor ownership → rational bubble) generalizes to other developed markets (UK, Canada, Japan, Eurozone) with similar magnitude and dynamics. However, the elasticity is modulated by: (i) market structure (concentrated index universes vs. diversified), (ii) regulatory environment (restrictions on index funds, tax efficiency), (iii) financial development (depth of market, short-selling ease). Predictions: largest effects in UK and Canada (highly indexed markets similar to US); smaller in Japan and Eurozone (lower passive market share, different index culture).

**Gap Addressed:** 
Jappelli's data is US-only. International evidence would validate generalizability of the mechanism and identify boundary conditions (which market features are essential for the bubble to exist). Also tests whether the mechanism is US-specific (e.g., due to unique regulatory features like tax-deferred accounts) or universal.

**Method:**

**Parallel analysis to Experiment A2 (aggregate price pressure) for multiple countries:**

For each country (UK, Canada, Japan, Germany, France), estimate:
$$\Delta P_{t}^{(c)} = \alpha^{(c)} + \beta^{(c)} \cdot \frac{\Delta F^{\text{static},c}_t}{MV^{(c)}_t} + \gamma^{(c)} \mathbf{Z}^{(c)}_t + \varepsilon^{(c)}_t$$

Where superscript (c) indexes country. Test H0: $\beta^{(c)} = \beta^{\text{US}}$ for each country. Collect estimates in a table and analyze heterogeneity.

**Cross-country regression to explain elasticity heterogeneity:**
$$\beta^{(c)} = \theta_0 + \theta_1 \cdot \text{Passive Share}^{(c)} + \theta_2 \cdot \text{Index Concentration}^{(c)} + \theta_3 \cdot \text{Short-Sale Ease}^{(c)} + u^{(c)}$$

Where:
- Passive Share = % of market in passive funds (varies by country)
- Index Concentration = Herfindahl index of top 10 stocks / total market cap (US high due to tech concentration)
- Short-Sale Ease = survey or regulatory measure of ease of short-selling (proxy for arbitrage limits)

**Time-series comparison:** For each country, estimate Experiment C1 (bad-times amplification) and compare coefficients; test whether amplification is universal or country-specific.

**Data Requirements:**
- Equity prices and returns: UK (LSE), Canada (TSX), Japan (Nikkei), Eurozone (EURO STOXX)
  - Source: Bloomberg, MSCI, national stock exchanges
  - Frequency: monthly, 2005–2024 (or available period)
- Passive fund flows by country: 
  - Source: fund databases (Morningstar, Bloomberg, national regulators)
  - Challenge: not all countries have unified public fund reporting; may require industry surveys or Lipper/Morningstar data
- Market cap and dividend data: same sources
- Regulatory indicators:
  - Passive fund market share: Vanguard reports, Morningstar
  - Index concentration: calculated from exchange data
  - Short-sale regulations: BlackRock ShortsightedNess index or regulatory reviews
- Currency-adjusted returns (if comparing in USD terms)

**Identification Strategy:**
- **Same IV approach as Experiment A2:** Use index rebalancing dates (international index providers: FTSE, TSX, Nikkei 225) as instruments for passive flows. Validity assumption: rebalancing dates are exogenous to contemporaneous price shocks (verify for each index).
- **Confounding by monetary policy:** Central banks' quantitative easing / tightening may affect equity prices in all countries. Control for policy rates, credit spreads, currency movements.
- **Data availability constraint:** Some countries have poorer reporting of fund flows and characteristics. Mitigate by using official central bank / regulatory statistics where available, supplemented by industry surveys.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| β^(c) | Price elasticity to passive flows, by country | Estimated |
| Passive Share | Passive fund AUM / total market cap (%) | Morningstar, regulators |
| Index Concentration | HHI of top 10 stocks / total market cap | Exchange data |
| Short-Sale Ease | Regulatory ease score (0–1 scale) | Regulatory reviews |
| Real Rates | Country policy rate − inflation expectations | Central banks |
| Currency Basis | FX forward premium vs. USD | Bloomberg |

**Robustness Checks:**
1. **Currency adjustment:** Verify results robust to USD vs. local-currency returns.
2. **Index definition:** Use country-specific indices (TOPIX for Japan, CAC40 for France) vs. international indices (MSCI); verify elasticity consistent.
3. **Subsample by period:** Run analysis pre-2008, 2008–2015, post-2015 separately; verify elasticity trends over time (should be increasing as passive share grows).
4. **Exclude crisis periods:** Drop months with major currency crises, sovereign debt crises; verify elasticity not driven by outliers.
5. **Placebo test:** Estimate elasticity for non-existent "futures flows" or "warrant flows"; should be zero or near-zero (falsification).

**Statistical Power:**
- Sample per country: N ≈ 200–280 months (depending on data availability)
- Cross-country: K = 5 countries, so meta-analysis of 5 estimates
- MDE for elasticity heterogeneity: 50% difference between countries (e.g., β_US = 3 bps, β_Japan = 1.5 bps) detectable at 70% power
- Cross-country regression: N = 5 (underpowered!), but descriptive

**Timeline:** 14 months (long data compilation phase)

**Expected Contribution:**
Validates the static-investor bubble mechanism internationally, showing it is not US-specific. Identifies market structure features that amplify or attenuate the effect. Implications for: (i) global asset managers (understanding price impact across markets), (ii) international policymakers (systemic risk from passive investing in their markets), (iii) comparative market microstructure (why do some markets have larger bubbles?).

**Risk Assessment:**
- **Data fragmentation:** International fund data is not centralized; significant effort required to harmonize definitions and obtain consistent time series. Plan 2–3 months for data compilation alone.
- **Index differences:** Different countries' primary indices have different structures (concentration, liquidity, turnover). May explain elasticity differences, but complicates interpretation.
- **Regime shifts:** Brexit, Eurozone crisis, COVID, etc. create structural breaks in each country's markets. Subsample analysis required.
- **Currency confounding:** If analyzing in USD, currency movements may affect results independently. Necessary to adjust.

---

## Experiment E2: Stress-Testing the Rational Bubble: Non-Fundamental Shocks and Market Crashes

**Type:** Stress-Test

**Hypothesis:** 
The rational bubble mechanism is fragile: when the Sharpe ratio drops below a critical threshold (e.g., −0.5, reflecting severe market stress), the martingale property breaks down and the bubble component collapses rapidly, amplifying the market decline. Mechanically: a 10% market decline in normal times translates to 12–15% decline if it coincides with negative Sharpe ratio shock and passive investor losses (due to the amplification in Experiment C1). This represents a "amplification multiplier" of 1.2–1.5× for passive-heavy market shocks.

**Gap Addressed:** 
Jappelli's theory assumes the martingale property holds even in bad times. Empirically, extreme stress (March 2020, 2008 crisis) may violate this, causing rapid bubble collapse. Understanding stress-time bubble dynamics is crucial for assessing systemic risk from passive investing. This experiment tests whether the rational bubble is robust to extreme shocks or becomes irrational in crises.

**Method:**

**Event-study approach around major market crises:**

Identify major market shocks (−5% single-day decline): 
- 1987 Black Monday (−22%)
- 2008 Lehman collapse
- 2011 US downgrade
- 2020 COVID crash (−12% in one day)
- Others identified from daily CRSP returns (e.g., percentile < 1st)

For each event, estimate:
1. **Bubble component decay:** Compare actual price decline to predicted decline (based on dividend fundamentals). If bubble collapses, residual V should decline faster than predicted by martingale.

   $$V_t^{\text{actual}} - V_t^{\text{model predicted}} = \delta \cdot \mathbb{1}(\text{crisis day}) + \text{higher-order terms}$$

   Test H0: δ = 0 (bubble unchanged) vs. HA: δ ≠ 0 (bubble responds to crisis).

2. **Amplification multiplier:** Regress total market decline on dividend-based predicted decline; slope = amplification. If slope > 1, bubble is amplifying the decline.

   $$\Delta P_{\text{total}} = \alpha + \beta \cdot \Delta P_{\text{fundamental}} + \varepsilon$$

   Test if slope in crisis periods > slope in normal periods.

**Time-series simulation with regime-switching:**

Build a structural model where:
- Fundamental dividend process: standard, exogenous
- Dynamic investor demand: Merton CRRA utility, responds to Sharpe ratio
- Static investor demand: fixed allocation θ
- Bubble component: martingale in normal times, but "jumps" if Sharpe ratio triggers a threshold

Simulate market paths under this model, comparing:
- Baseline (no bubble): only fundamental + dynamic investor demand
- With-bubble (normal times): fundamental + dynamic + martingale bubble
- With-bubble (crisis): fundamental + dynamic + bubble collapse when Sharpe < threshold

Compare simulated crash magnitudes to actual; verify that adding bubble amplifies crashes.

**Stress-test scenarios:** 
- Scenario A: 50% of market suddenly forced to liquidate (e.g., severe redemption wave)
- Scenario B: Credit crunch (borrowing costs for dynamic investors spike)
- Scenario C: VIX shock (risk aversion spike)

For each scenario, predict price decline with and without static investor presence. Difference = systemic risk from static investors.

**Data Requirements:**
- High-frequency price data (daily CRSP): 1980–2024, especially for crisis periods
- Daily/intraday volume and bid-ask spreads (if available from CRSP or TAQ) to estimate liquidity constraints
- Realized Sharpe ratio on crisis days
- Fund redemption flows during crises (if data available)
- Dividend expectations and actual dividend announcements around crises

**Identification Strategy:**
- **Exogeneity of crisis dates:** Major crises are exogenous to US stock market (e.g., 1987 crash was international origin; 2008 was banking system failure; COVID was pandemic). Use as quasi-natural experiments.
- **Confounding by information:** Crises may contain new information about fundamentals. Mitigate by conditioning on contemporaneous earnings surprises, analyst revisions (should be small). If large fundamental surprises coincide with crashes, harder to interpret bubble collapse.
- **Model misspecification:** Structural model is simplified; actual market more complex. Use simulation for qualitative (does bubble amplify?) rather than quantitative (by exactly how much?) predictions.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| Daily returns | Daily log return, S&P 500 or CRSP VW index | CRSP |
| Daily fundamentals | Expected dividend yield from VAR model | Estimated from CRSP |
| Daily Sharpe ratio | Realized (return − rf) / intraday volatility | Calculated |
| Fund flows | Daily estimated flows (from volume/positioning data) | CRSP, reconstructed |
| VIX | Intraday implied volatility | CBOE |
| Bid-ask spread | Daily average (if available) | CRSP/TAQ |

**Robustness Checks:**
1. **Alternative crisis definitions:** Use different thresholds (−3%, −5%, −10%) to identify events; verify amplification effect present for all.
2. **Sample-period sensitivity:** Rerun for subsets (pre-2000, post-2000, post-2010) as passive share grew; verify amplification larger as passive grows.
3. **Alternative fundamental models:** Replace VAR-based PDV with Gordon-growth model or dividend discount model; verify results robust.
4. **Model parameters:** Vary CRRA coefficients, static allocation shares in simulation; verify amplification is not driven by extreme parameter choices.
5. **Falsification tests:** Estimate amplification for (i) fictitious "crashes" (random days, not actual crashes), should be zero; (ii) large positive returns, test whether bubble attenuates booms (opposite of amplifying crashes).

**Statistical Power:**
- Sample: K ≈ 15–20 major crisis events in 1980–2024
- Effect size: amplification multiplier 1.2–1.5× (10–50% increase in crash magnitude)
- Power: Modest (small K) but qualitative trends may still be evident

**Timeline:** 10 months

**Expected Contribution:**
Assesses systemic risk from passive investing, showing whether the rational bubble mechanism breaks down in crises (amplifying crashes). Provides regulators with evidence on severity of passive investor procyclicality. Useful for: (i) stress-testing frameworks (passive share as a risk factor), (ii) policy design (should passive investing be regulated?), (iii) risk management (investors and funds can forecast crash severity given passive share).

**Risk Assessment:**
- **Sample size of crises:** Only 15–20 major events; limited power for statistical inference. Mitigate by combining with simulation and cross-country evidence (more crises in international data).
- **Model misspecification:** Structural models always simplified. Focus on qualitative (directional) predictions rather than point estimates.
- **Regime-switching complexity:** Real behavior in crises is complex; simplified model may miss key mechanisms (e.g., forced selling, liquidity spirals). Use simulation as a thought experiment, not a definitive forecast.
- **Confounding by policy responses:** Major crises trigger central bank interventions; difficult to isolate private-sector bubble dynamics from policy effects. Subtract out estimated policy impact.

---

## Experiment E3: Portfolio Rebalancing Behavior and Passive Investor Demand Elasticity

**Type:** Extend

**Hypothesis:** 
Static investors' price responsiveness depends on their rebalancing policy. Funds that rebalance mechanically on fixed calendars (e.g., annually, semi-annually) create inelastic demand (buy/sell regardless of price), amplifying prices. Funds that rebalance when allocation drifts beyond a threshold (e.g., ±5% band) create more elastic demand (less buying/selling if prices already high). Empirically: calendar-rebalancers show price elasticity β ≈ 2–3 bps per 0.1% flow; threshold-rebalancers show β ≈ 0.5–1 bps (3–6× smaller). Time-varying elasticity based on observed rebalancing frequency explains heterogeneity in past flow-impact studies.

**Gap Addressed:** 
Jappelli treats static investors as having fixed θ (allocation), but in reality, rebalancing strategies vary. Some funds (e.g., target-date) rebalance more frequently as retirement date approaches; others (e.g., some balanced funds) rebalance only annually or when drift is large. This variation is not studied, but may be crucial for understanding price impact heterogeneity and whether the bubble mechanism is robust to different rebalancing rules.

**Method:**

**Classify funds by rebalancing rule:**

Use fund prospectus or CRSP data (if available) to classify:
- **Type 1 (Calendar):** Rebalance on fixed schedule (typically semi-annually or annually). Examples: many target-date funds, some balanced funds.
- **Type 2 (Threshold):** Rebalance when allocations drift beyond target ±k% (k typically 5–10). Examples: some balanced funds, lifestyle funds.
- **Type 3 (Constant Proportion):** Use floor/ceiling rules (e.g., always maintain 60±5% equities). Examples: some managed-account platforms.

**Separate analysis for each rebalancing type:**
$$\Delta P_t = \alpha^{(j)} + \beta^{(j)} \cdot \frac{\Delta F^{(j)}_t}{MV_t} + \gamma^{(j)} \mathbf{Z}_t + \varepsilon_t^{(j)}$$

Where j ∈ {1, 2, 3}. Estimate $\beta^{(j)}$ for each type; test if they differ significantly.

**Endogeneity within rebalancing type:**

Instrument using exogenous variation in drift (e.g., large market move triggers rebalancing for threshold-rebalancers, but not for calendar-rebalancers). 

For calendar-rebalancers: use month/quarter dummies to predict when flows will be large (exogenous to prices, conditional on calendar).

For threshold-rebalancers: use lagged market moves to predict current drift, then instrument flows with drift.

**Time-varying elasticity based on drift:**

Estimate:
$$\Delta P_t = \alpha + \beta(\text{Drift}_t) \cdot \frac{\Delta F_t}{MV_t} + \varepsilon_t$$

Where Drift_t is estimated distance from target allocation, and β is a function of drift (e.g., beta increases when drift is large / rebalancing imminent, using local linear regression).

**Prospectus analysis (qualitative):**

Manually review fund prospectuses for ~200–300 target-date and balanced funds to code rebalancing rules. Supplement with CRSP classification if available.

**Data Requirements:**
- Fund holdings (quarterly): derive drift = (current allocation − target allocation) for each fund
- Fund portfolio turnover and trade dates (from holdings or prospectus)
- Fund flows and returns (CRSP Mutual Fund Summary): monthly or quarterly
- Equity market returns and market cap (CRSP)
- Fund prospectuses (SEC EDGAR, fund websites)

**Identification Strategy:**
- **Exogeneity of rebalancing rule:** Assume fund's choice of rebalancing rule is predetermined (made at fund inception, rarely changed). Verify by checking for rule changes in fund history.
- **Calendar exogeneity:** Assume rebalancing months (e.g., December for annual rebalancers) are exogenous to contemporaneous market shocks. Test by verifying December flows are similar in magnitude across years.
- **Drift exogeneity:** For threshold-rebalancers, assume drift is due to past market movements (exogenous), not endogenous response to current market. Verify by regressing drift on lagged returns (not contemporaneous).

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| Rebalancing Type | 1=calendar, 2=threshold, 3=constant prop | Prospectus |
| Drift | Current allocation − target allocation (%) | Calculated from holdings |
| Flow | Net new assets, by fund | CRSP MF Summary |
| Turnover | Annual portfolio turnover ratio | CRSP |
| Trade frequency | Implied from holdings changes | Calculated |

**Robustness Checks:**
1. **Alternative drift measures:** Use (i) absolute drift (|drift|), (ii) drift/volatility (normalized), (iii) extreme drift (binary: >5% vs. ≤5%).
2. **Placebo test:** Estimate elasticity by rebalancing rule for passive benchmark funds (should show no elasticity, regardless of rule). If passive funds show zero elasticity by rule, validates identification.
3. **Interaction with market conditions:** Test if elasticity differences are larger in high-volatility periods (when rebalancing pressure is greater).
4. **Fund-level vs. aggregate:** Repeat analysis at aggregate level (all Type 1 flows combined vs. all Type 2 flows) and at fund-level; verify results consistent.

**Statistical Power:**
- Sample: ~2,000–3,000 target-date and balanced funds, quarterly observations, 2005–2024 = ~2M fund-quarters
- Funds distributed across types: 40% calendar, 40% threshold, 20% other
- MDE for elasticity differences: 50% variation across types (e.g., β_cal = 2 bps, β_thresh = 1 bps) detectable at 85%+ power

**Timeline:** 11 months

**Expected Contribution:**
Reveals that static investor price impact depends on rebalancing rule, explaining variation in elasticity across fund families and periods. Suggests that fund design (rebalancing policy) has externalities (affect market prices). Implications for: (i) fund managers (choosing rebalancing policy affects market impact and investor experience), (ii) academic literature (need to account for rebalancing heterogeneity when studying passive investor effects), (iii) policy (potential role for regulating rebalancing rules to mitigate price pressure).

**Risk Assessment:**
- **Prospectus analysis is labor-intensive:** Requires hand-coding 200–300 prospectuses. Mitigate by using subset or collaborating with research assistant.
- **Rebalancing rule inference from holdings:** Some rules are not explicit; must infer from observed trading patterns. Risk of misclassification. Mitigate by triangulating with prospectus data and CRSP classification.
- **Small subsample size for some types:** If Type 3 funds are rare, analysis of that group is underpowered. Focus on Types 1 and 2 (most common).

---

## Experiment E4: Hidden Bubbles in Sector and Factor Exposures

**Type:** Extend

**Hypothesis:** 
The rational bubble mechanism extends beyond price levels to factor exposures and sector allocations. Sectors with high static investor concentration (e.g., Tech through S&P 500 concentration) or high correlation with the market (low beta heterogeneity) experience larger bubbles. Quantitatively: Fama-French factor exposures (especially market beta) of static funds create factor-specific bubbles. E.g., the market factor has a bubble component (from static investors' systematic long exposure), but specific factors (value, momentum, low-vol) may have negative bubbles (because static investors underweight these factors via index investing). Empirically: value and momentum factors underperform their predicted returns by 2–4% annualized due to negative static investor exposure.

**Gap Addressed:** 
Jappelli's aggregate and cross-sectional results are about individual stocks and aggregate price level; factor-level bubbles are unexplored. Understanding factor-specific bubbles clarifies whether the static investor effect is a "broad market lift" or a "reweighting across risks." This has implications for active managers (value and momentum strategies may benefit from sector rotation away from high-static-ownership sectors).

**Method:**

**Factor-level price pressure regression:**

Define factor-mimicking portfolios (long stocks high on factor, short low). For each factor (market, size, value, momentum, quality, low-vol):
$$\text{Factor Return}_t = \alpha + \beta_1 \cdot \text{Static Exp}^{\text{Factor}}_{t-1} + \text{controls} + \varepsilon_t$$

Where Static Exp^Factor is the weighted exposure of static investors to that factor (computed as: Σ static ownership × factor loading of each stock).

**Cross-sectional decomposition:**

Regress stock-level returns on:
$$r_{it} = \alpha_t + \sum_f \beta_{it}^{(f)} \cdot \text{Factor Return}_{t}^{(f)} + \text{Alpha}_t + \varepsilon_{it}$$

Decompose Alpha into:
$$\text{Alpha}_{it} = \text{Factor Bubble}_{it} + \text{Stock-Specific Bubble}_{it}$$

Where Factor Bubble depends on stock's factor loadings and static fund factor exposures, and Stock-Specific Bubble is the residual.

**VAR on factor flows and returns:**

Extend the VAR from Experiment C2 to include multiple factors:
$$\mathbf{X}_t = [r_t^{\text{mkt}}, r_t^{\text{value}}, r_t^{\text{momentum}}, F^{\text{static,mkt}}_t, F^{\text{static,value}}_t, ...]'$$

Estimate impulse responses of each factor return to shocks in static fund flows to that factor. Hypothesis: market factor shows strong positive response (large bubble), value and momentum show zero or negative response (small bubble or negative bubble due to underweighting).

**Sector-level analysis (special case):**

Define sector exposures from static fund holdings. Test whether sectors with high static ownership (e.g., large-cap tech in S&P 500) have lower expected returns controlling for risk factors. Use double-sort: by static ownership and by sector; estimate long-short return.

**Data Requirements:**
- Stock returns and characteristics (Compustat/CRSP): 1980–2024
- Factor returns (Fama-French library): market, SMB, HML, RMW, CMA, Momentum
- Factor loadings: estimated via rolling regressions or from factor model
- Fund holdings (quarterly): to compute static fund factor exposures
- Sector classifications (GICS or SIC): to map stocks to sectors

**Identification Strategy:**
- **Factor exposure exogeneity:** Assume static fund exposure to a factor is driven by index construction rules, not endogenous reweighting. E.g., if S&P 500 is static fund benchmark, value/momentum underweighting is mechanical (index just doesn't weight these factors highly). Verify by showing static fund factor exposures are stable over time.
- **Factor loading stability:** Assume stock factor loadings (beta, value loading, etc.) are stable or slowly changing. Estimate via rolling regressions; verify loadings not correlated with shocks.

**Key Variables:**

| Variable | Definition | Source |
|----------|-----------|--------|
| Factor Return | Fama-French factor return (monthly or daily) | Ken French Library |
| Static Exp^Factor | Σ(SO_i × β_i^Factor), exposure of static funds to factor | Calculated |
| Sector Return | Value-weighted return of stocks in sector | CRSP |
| Sector SO | Average static ownership in sector | Calculated |
| Factor Loading | Stock exposure to factor (beta, value, momentum) | Estimated from CRSP |

**Robustness Checks:**
1. **Alternative factor definitions:** Use (i) Fama-French 5-factor, (ii) other factors from literature (profitability, investment), (iii) style factors (growth, value).
2. **Subsample by factor exposure:** Repeat for stocks high vs. low in each factor; verify static ownership effect concentrated in high-factor stocks.
3. **Interaction with macro:** Test whether factor bubble sizes vary with Sharpe ratio (as in Experiment C1, but at factor level).
4. **Falsification:** Estimate bubbles for factors static funds have NO exposure to (e.g., short-only momentum strategies); should be zero.

**Statistical Power:**
- Sample: 5 main factors × 20 years × 12 months = 1,200 obs per factor
- Effect size: factor return 1–3 bps per 0.1% static exposure change; MDE ≈ 0.5 bps
- Power: 80%+ per factor

**Timeline:** 9 months

**Expected Contribution:**
Extends the bubble mechanism from aggregate and stock level to factor level, revealing hidden reweighting effects. Shows that passive investing's impact on markets is more subtle than "broad lift" — it creates factor-specific bubbles and mispricings. Useful for: (i) active managers (identifying factors that are undervalued due to passive underweighting), (ii) academics (explaining factor return puzzles, e.g., value factor underperformance), (iii) policymakers (understanding which investors bear the cost of passive investing's market impact).

**Risk Assessment:**
- **Factor collinearity:** Factors are correlated (SMB and HML loadings correlated with market beta). Multicollinearity may inflate standard errors. Mitigate by using orthogonalized factors or principal components.
- **Time-varying loadings:** Stock betas and factor loadings vary over time. May create spurious correlations if not accounted for. Use rolling regressions and control for lagged loadings.
- **Multiple testing:** With 5+ factors, risk of false discovery. Apply FDR correction or pre-register hypotheses.

---

# EXPERIMENT SUMMARY TABLE

| Block | Exp | Title | Type | Timeline | Key Innovation |
|-------|-----|-------|------|----------|-----------------|
| **A** | A1 | Direct Test of Rational Bubble | Validate | 6 mo | Martingale property test, transversality violation |
| | A2 | Aggregate Price Pressure from Static Inflows | Validate | 8 mo | Flow elasticity measurement with IV |
| **B** | B1 | Static Ownership and Cross-Sectional Returns | Validate | 10 mo | Fama-MacBeth with 20 years data |
| | B2 | Heterogeneous Effects by Firm Characteristics | Extend | 8 mo | Interaction analysis, causal forest methods |
| **C** | C1 | Bad-Times Amplification | Validate | 9 mo | Time-varying parameters, Sharpe ratio interaction |
| | C2 | Inflow-Return Causality and Dynamics | Extend | 10 mo | VAR/impulse responses, local projections |
| **D** | D1 | Static vs. Dynamic Flow Elasticity | Validate | 11 mo | Differential price impact estimation |
| | D2 | State-Dependent Elasticity and Spillovers | Extend | 12 mo | Multi-asset spillovers, regime-dependence |
| **E** | E1 | International Validation | Extend | 14 mo | Cross-country comparison, market structure effects |
| | E2 | Stress-Testing and Crisis Dynamics | Stress | 10 mo | Event studies, structural simulation |
| | E3 | Portfolio Rebalancing Behavior | Extend | 11 mo | Prospectus analysis, rebalancing rule heterogeneity |
| | E4 | Factor-Level Bubbles | Extend | 9 mo | Fama-French factor decomposition |

---

# INTEGRATION WITH PROPRIETARY DATA ASSETS

## Haddad et al. (2025) Demand System Estimates

**Leverage points:**
- **Experiments A1–A2, B1:** Use estimated elasticities from demand system to predict price impact directly; compare to reduced-form estimates for validation.
- **Experiment D1:** Use demand system estimates to decompose static vs. dynamic elasticities; test whether decomposition matches econometric estimates.
- **Experiment E2:** Use demand-system cross-asset elasticities to predict spillovers; validate against observed spillovers.
- **Experiment E4:** Use demand system estimates of factor-specific elasticities to predict factor bubbles; test predictions empirically.

**Integration strategy:** 
Treat Haddad estimates as structural parameters; derive theoretical predictions and test against empirical estimates. Reconcile any discrepancies (may reveal model misspecification or demand-system parameter uncertainty).

---

## Jiang et al. (2025) Index Data and Flow IVs

**Leverage points:**
- **Experiments A2, B1, D1:** Use S&P 500/600 index inclusion as instrument for static fund flows (exogenous demand shifter).
- **Experiment E1:** Use index rebalancing dates from multiple indices (FTSE, TSX, etc.) as country-specific instruments for international analysis.
- **Experiment E3:** Use index composition changes to identify shocks to factor exposures; test whether these create factor-specific price pressure.
- **Experiment C2:** Use index reconstitution timing in VAR to improve identification of causal dynamics.

**Integration strategy:**
Index inclusion/exclusion is a powerful quasi-natural experiment for isolating exogenous flows. Use flow IVs to construct strong instruments, ensuring causal inference in reduced-form regressions and VAR analysis.

---

# PROJECT MANAGEMENT AND TIMELINE

## Phase 1: Foundation (Months 1–6)

**Quick-win experiments (Validate):**
- A1: Direct test of rational bubble (6 mo)
- A2: Aggregate price pressure (8 mo, starts Mo 1, completes Mo 8)
- C1: Bad-times amplification (9 mo, starts Mo 1)

**Parallel data work:**
- Compile CRSP mutual fund data (holdings, flows, allocations) through 2024
- Construct static/dynamic fund classifications; validate against papers
- Calculate Haddad and Jiang datasets into usable form

**Deliverables:**
- Technical report on Proposition 1 (martingale property, transversality)
- Flow-elasticity estimates for aggregate market

## Phase 2: Cross-Sectional and Decomposition (Months 6–14)

**Main experiments:**
- B1: Stock-level returns (10 mo, starts Mo 4)
- B2: Heterogeneous effects (8 mo, starts Mo 6)
- D1: Static vs. dynamic elasticity (11 mo, starts Mo 4)
- C2: Inflow-return causality (10 mo, starts Mo 6)
- E3: Rebalancing behavior (11 mo, starts Mo 5)

**Deliverables:**
- Portfolio sort paper on Proposition 2
- Flow-impact decomposition (static vs. dynamic)
- VAR analysis of price dynamics

## Phase 3: Extensions and Stress Tests (Months 14–24)

**Ambitious experiments:**
- E1: International validation (14 mo, starts Mo 2 for long data compilation)
- E2: Stress-testing (10 mo, starts Mo 13)
- E4: Factor bubbles (9 mo, starts Mo 15)
- D2: Spillovers (12 mo, starts Mo 10)

**Deliverables:**
- International evidence paper
- Systemic risk assessment (crisis amplification)
- Factor-level analysis

---

# RISKS AND MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| **Data quality/gaps** | High | High | Start data compilation immediately; use Haddad/Jiang as backup; have fallback analysis plan for limited samples |
| **Endogeneity (reverse causality)** | High | High | Rely on IV and quasi-natural experiments; Granger causality tests; falsification tests |
| **Multiple testing/false discovery** | Medium | Medium | Pre-register hypotheses; use FDR correction; report all tests |
| **Structural breaks (passive investing boom)** | Medium | Medium | Subsample analysis pre/post 2008, 2010; test regime-switching models |
| **Model misspecification** | Medium | Medium | Use multiple specifications; robustness checks; avoid overinterpreting point estimates |
| **International data scarcity** | High | Medium | Focus on 2 main countries (UK, Canada) if full 5-country sample infeasible |
| **Stress-test sample size** | Medium | Low | Combine event-study with simulation; qualitative discussion acceptable |

---

# EXPECTED OUTPUTS AND PUBLICATIONS

1. **Main paper:** "Asset Pricing with Static and Dynamic Investors: Empirical Validation of the Rational Bubble Mechanism" (Experiments A1–A2, B1–B2, C1–C2). Target: top-5 finance journal.

2. **Second paper:** "Flow Decomposition and Market Elasticity: Static vs. Dynamic Investors" (Experiments D1–D2). Target: market microstructure journal.

3. **Third paper:** "Passive Investing and Systemic Risk: International Evidence and Stress Testing" (Experiments E1–E2). Target: macrofinance journal or top-5.

4. **Fourth paper:** "Portfolio Rebalancing, Factor Bubbles, and Price Pressure" (Experiments E3–E4). Target: asset pricing journal.

5. **Supplementary materials:** Working papers with detailed robustness checks, international appendices, and additional analysis.

---

# CONCLUSION

This comprehensive agenda of 12 experiments is designed to thoroughly validate, extend, and stress-test Jappelli (2025)'s theoretical framework. By leveraging proprietary data from Haddad et al. (2025) and Jiang et al. (2025), the research will provide robust empirical evidence on the mechanisms through which static investors create price pressure. The mix of validation, extension, and stress-testing ensures both rigor and novelty, positioning the research for high-impact publication and meaningful contribution to asset pricing theory and market microstructure.