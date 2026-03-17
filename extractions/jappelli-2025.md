# Extraction Card: Asset Pricing with Dynamic and Static Investors

## Bibliographic Info
- **Authors**: Jappelli, R.
- **Year**: 2025
- **Title**: Asset Pricing with Dynamic and Static Investors
- **Journal/Source**: SSRN Working Paper No. 6168736 (Warwick Business School)
- **DOI**: SSRN-6168736
- **APA Citation**: Jappelli, R. (2025). Asset pricing with dynamic and static investors. *SSRN Working Paper*. https://ssrn.com/abstract=6168736

## Journal Prestige Weight
- **Tier**: Working paper
- **Rationale**: SSRN working paper with strong institutional affiliation (University of Warwick, Warwick Business School). Awarded the John A. Doukas best Ph.D. paper at the 2024 European Financial Management Association Conference and best Ph.D. paper at the 6th Asset Pricing Conference (Long-Term Investors, Collegio Carlo Alberto). Presented at CEPR Paris Symposium 2025, AFA Junior Faculty Mentoring Workshop, and multiple top universities. Trajectory suggests top-tier journal target (JF/JFE/RFS).

## Core Thesis
The paper develops a closed-form Intertemporal Capital Asset Pricing Model (ICAPM) with two heterogeneous investor types: dynamic investors who optimize portfolios in response to news (Merton, 1973) and static investors (balanced mutual funds, ETFs) who maintain fixed asset allocation targets regardless of market conditions. The central result is that the equilibrium price of equity equals fundamentals (discounted dividends) PLUS the equity exposure of static investors (θV_t), constituting a "rational bubble" that is fully compatible with no-arbitrage because dynamic investors cannot profitably trade against it. This "Asset Classification Effect" means aggregate equity valuations depend on the level of wealth managed under static strategies, above and beyond dividends, discount rates, and flows.

## Methodology
- **Approach**: Theoretical (with empirical quantification)
- **Data**: CRSP Survivor-Bias-Free Mutual Fund Database (Summary Files), quarterly 2004:Q1–2024:Q4. Funds classified as "static" if standard deviation of equity allocation ≤ 5% over sample period. Fund-level variables computed as TNA-weighted averages of underlying share classes. Sample excludes funds with TNA < $5M, non-U.S.-equity objective codes, and allocation anomalies outside [75%, 125%].
- **Methods**:
  - Continuous-time pure-exchange economy with infinite horizon
  - CARA utility for dynamic investors (baseline); CRRA robustness in Appendix E
  - Hamilton-Jacobi-Bellman equation for dynamic investors' optimization
  - Constrained mean-variance optimization for static stock pickers
  - Closed-form equilibrium price via ansatz-and-verification
  - Stochastic Discount Factor estimation using Fama-French 5 factors (2015)
  - Martingale restriction test with Newey-West standard errors (4 quarterly lags)
- **Identification Strategy**: Theoretical model with closed-form solution. Empirical section uses the model's testable restriction: θV_t must be a risk-adjusted martingale (Remark 1). Tests this unconditional moment condition E_t[M_t · θV_t − θV_{t-1}] = 0.
- **Robustness Checks**:
  1. Correlation between wealth flows and earnings news (Internet Appendix B)
  2. CRRA utility for dynamic investors (Appendix E) — asset classification effect persists
  3. Microfounding static investors via observation-cost model (Appendix B)
  4. Flexible asset allocation targets (Buffa et al., 2022 framework)
  5. Wealth flows with drift term (Appendix D)
  6. Survival of agents — price impact depends on wealth level, not wealth share (Kogan et al., 2006)

## Key Findings
1. **Asset Classification Effect (Proposition 1)**: The equilibrium price of the equity asset class equals P_t = PDV_t(D_t) + θV_t, where θV_t is the equity exposure of static investors. This term constitutes a rational bubble — the price exceeds the discounted dividend stream without creating arbitrage opportunities.
2. **No-Arbitrage Compatibility**: Static investors' discounted equity exposure is a risk-adjusted martingale (Remark 1), so dynamic investors cannot profitably short-sell against the price pressure. The transversality condition is violated (Remark 2), but this is rational given the infinite-horizon and perpetual predictability of static demand.
3. **Time-Series Variation**: Static investors' wealth becomes relatively more important for prices when expected risk-adjusted returns on stocks are low (bad times), because dynamic investors shift to bonds. The Sharpe ratio and static fund ownership share exhibit a correlation of −0.21 in levels and −0.65 in first differences.
4. **Cross-Section (Proposition 2)**: Individual stock prices reflect P_it = PDV(D_it) + θ[q*_it · V^A_t + λ_it · V^P_t]. Index stocks and stocks with favorable risk-return profiles experience the largest valuation effects. Both index trackers AND stock pickers exert aggregate price pressure.
5. **Empirical Quantification**: Static funds' equity exposure grew from ~$2T (2004) to ~$6.6T (2024), while U.S. market cap grew from $14.3T to $57.4T. Static funds hold ~14.5% of the U.S. stock market on average. The martingale test cannot reject the null (p-value = 0.32).

## Limitations
### Noted by author:
- Model assumes constant risk-free rate (exogenous bond supply)
- CARA utility baseline prevents wealth effects (addressed via CRRA robustness)
- Static allocation targets are exogenous (partially addressed via microfounding in Appendix B)
- Cross-sectional analysis abstracts from portfolio constraints on dynamic investors
- Empirical section is a quantification exercise, not a formal causal test of the asset classification effect

### Additional limitations identified:
- Classification of funds as "static" (SD ≤ 5%) is somewhat arbitrary; sensitivity to this threshold not fully explored
- Model assumes homogeneous risk aversion across dynamic and static stock-picker agents
- Only U.S. equity market; international applicability untested
- Does not model transitions between static and dynamic strategies over time (entry/exit)
- Bond market treated as exogenous; no feedback from equity price pressure to interest rates
- No explicit consideration of short-selling costs or other market microstructure frictions beyond the theoretical frictionless setup
- Limited empirical testing — primarily one moment condition test; no cross-sectional empirical verification of Proposition 2

## Connections to Other Papers
### Direct theoretical predecessors:
- **Merton (1973)**: Dynamic portfolio choice framework for dynamic investors
- **Gabaix and Koijen (2023)**: Inelastic markets hypothesis; static allocation as source of price pressure in Lucas tree model. Jappelli's paper departs by showing compatibility with dynamic arbitrageurs.
- **Tirole (1985)**: Rational bubble theory; asset classification effect fits within this framework
- **Veronesi (1999)**: CARA-utility continuous-time asset pricing with similar optimization program

### Methodological relatives:
- **Basak and Pavlova (2013)**: Institutional investors and benchmarking in equity markets
- **Chien et al. (2011, 2012)**: Heterogeneous asset allocation strategies matching asset prices and wealth distribution
- **Duarte et al. (2025)**: Active/passive investors in Lucas economy (independent parallel work)
- **Koijen and Yogo (2019)**: Demand system approach to asset pricing

### Empirical relatives:
- **Da et al. (2018)**: Wealth reallocation between funds with different targets generates aggregate price pressure
- **Edelen and Warner (2001)**, **Warther (1995)**: Mutual fund flows and stock prices
- **Hartzmark and Solomon (2025)**: Dividend distributions and price pressure
- **Harvey et al. (2025)**: Portfolio rebalancing from institutional investors with static strategies
- **Harris and Gurel (1986)**, **Shleifer (1986)**: Index inclusion effects (cross-sectional analog)
- **Dou et al. (2025)**: Cross-sectional sensitivity of stocks to aggregate mutual fund flows

### Extensions suggested by the author:
- Private equity and commodities asset classes
- Multiple indexes (Pavlova and Sikorskaya, 2023)
- Time trend in passive vs. active allocation (Jiang et al., 2025)
- Style investing and style labels (Boyer, 2011)

## Key Quotes
- "In equilibrium, the aggregate valuation of equity assets depends on the level of wealth managed by static investors, over and above dividends, discount rates, and the wealth flows they receive." (Abstract, p. 1)
- "A broader implication of the equilibrium is that the price impact of a trade depends on the intertemporal investment approach of its initiator." (p. 15)
- "Policymakers appear inclined to harness the effects described in this paper, as they often encourage investment vehicles under their influence to raise their domestic equity allocation targets in order to stimulate domestic financial conditions." (p. 24)

## Extraction Confidence
- **Parse Quality**: Full
- **Notes**: Complete text extraction across all 39 pages. All equations, propositions, proofs, and appendices readable. Figures and Table 1 successfully captured. Well-structured theoretical paper with clear notation and comprehensive appendices.
