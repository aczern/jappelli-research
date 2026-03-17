"""
Experiment Orchestrator: wires cached data to experiment execution.

Connects the data pipeline (WRDS downloads, FF factors, FRED series)
to the 12 Jappelli (2025) experiments via a phased, block-by-block
execution model.

Usage:
    python3 -m jappelli_experiments.orchestrator --block data   # Load + compute intermediates
    python3 -m jappelli_experiments.orchestrator --block a      # Run Block A
    python3 -m jappelli_experiments.orchestrator --block all    # Run everything
"""
import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Wires cached data -> intermediate variables -> experiment execution."""

    def __init__(self, skip_downloads=False, colab_mode=False, cache_dir=None,
                 output_dir=None):
        self.skip_downloads = skip_downloads
        self.colab_mode = colab_mode

        # Override CACHE_DIR globally if a custom path is provided
        if cache_dir is not None:
            import jappelli_experiments.config as cfg
            cfg.CACHE_DIR = Path(cache_dir)
            # Also update the cache module's reference
            import jappelli_experiments.data.cache as cache_mod
            cache_mod.CACHE_DIR = cfg.CACHE_DIR

        # Override output paths globally if a custom path is provided
        if output_dir is not None:
            import jappelli_experiments.config as cfg
            cfg.OUTPUT_DIR = Path(output_dir)
            cfg.TABLE_DIR = cfg.OUTPUT_DIR / "tables"
            cfg.FIGURE_DIR = cfg.OUTPUT_DIR / "figures"
            cfg.LOG_DIR = cfg.OUTPUT_DIR / "logs"
            cfg.INTERMEDIATE_DIR = cfg.OUTPUT_DIR / "intermediate"

            # Patch consumer modules' local name bindings
            import jappelli_experiments.shared.table_formatter as tf_mod
            tf_mod.TABLE_DIR = cfg.TABLE_DIR
            import jappelli_experiments.shared.plot_config as pc_mod
            pc_mod.FIGURE_DIR = cfg.FIGURE_DIR
            import jappelli_experiments.shared.connection_mapper as cm_mod
            cm_mod.INTERMEDIATE_DIR = cfg.INTERMEDIATE_DIR
            cm_mod.LOG_DIR = cfg.LOG_DIR

        # Raw data
        self.crsp_msi = None
        self.crsp_msf = None
        self.mf_summary = None
        self.compustat = None
        self.ccm_link = None
        self.ff_factors = None
        self.fred_data = None
        self.rf_rate = None
        self.sp500_events = None

        # Intermediates
        self.fund_class = None
        self.theta_t = None
        self.sharpe_t = None
        self.fund_flows = None
        self.aggregate_flows = None
        self.market_value = None
        self.aggregate_panel = None
        self.flow_pressure = None
        self.stock_panel = None
        self.so_it = None  # Set after Block B

        # Results
        self.results = {}

        # Phase tracking
        self._phase_0_done = False
        self._phase_1_done = False

        # Ensure output directories exist (use config module for patched values)
        import jappelli_experiments.config as cfg
        for d in [cfg.OUTPUT_DIR, cfg.INTERMEDIATE_DIR, cfg.LOG_DIR,
                  cfg.TABLE_DIR, cfg.FIGURE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Phase 0: Load all cached data ────────────────────────────

    def phase_0_load_data(self):
        """Load all cached parquets into memory."""
        logger.info("=" * 60)
        logger.info("PHASE 0: Loading cached data")
        logger.info("=" * 60)

        if self.colab_mode:
            self._load_all_from_cache()
        else:
            if not self.skip_downloads:
                self._download_free_data()
            self._load_with_downloads()

        # S&P 500 events (soft failure — file-based, may not exist in Colab)
        try:
            from jappelli_experiments.data.loaders import load_sp500_adddrop
            self.sp500_events = load_sp500_adddrop()
            logger.info(f"  sp500_events: {self.sp500_events.shape}")
        except (FileNotFoundError, Exception):
            logger.warning("  sp500_events: NOT AVAILABLE (file not found)")

        self._phase_0_done = True
        logger.info("PHASE 0 COMPLETE")

    def _load_with_downloads(self):
        """Load data via wrds_download / ff_download / fred_download (original path)."""
        from jappelli_experiments.data.wrds_download import (
            download_crsp_msi, download_crsp_msf, download_crsp_mf_summary,
            download_compustat_funda, download_ccm_link,
        )

        self.crsp_msi = download_crsp_msi(use_cache=True)
        logger.info(f"  crsp_msi: {self.crsp_msi.shape}")

        self.crsp_msf = download_crsp_msf(use_cache=True)
        logger.info(f"  crsp_msf: {self.crsp_msf.shape}")

        self.mf_summary = download_crsp_mf_summary(use_cache=True)
        logger.info(f"  mf_summary: {self.mf_summary.shape}")

        self.compustat = download_compustat_funda(use_cache=True)
        logger.info(f"  compustat: {self.compustat.shape}")

        self.ccm_link = download_ccm_link(use_cache=True)
        logger.info(f"  ccm_link: {self.ccm_link.shape}")

        # Fama-French
        from jappelli_experiments.data.ff_download import get_ff6_monthly
        self.ff_factors = get_ff6_monthly(use_cache=True)
        logger.info(f"  ff_factors: {self.ff_factors.shape}")

        # FRED
        from jappelli_experiments.data.fred_download import (
            get_fred_monthly, get_risk_free_rate,
        )
        self.fred_data = get_fred_monthly()
        if self.fred_data is not None:
            logger.info(f"  fred_data: {self.fred_data.shape}")
        else:
            logger.warning("  fred_data: NOT AVAILABLE")

        self.rf_rate = get_risk_free_rate("monthly")
        logger.info(f"  rf_rate: {len(self.rf_rate)} months")

    def _load_all_from_cache(self):
        """Load all data from parquet cache (no WRDS, no network).

        Bypasses wrds_download.py entirely so `import wrds` never executes.
        Used when colab_mode=True.
        """
        from jappelli_experiments.data.cache import load_cache
        import jappelli_experiments.config as cfg

        # ── CRSP + Compustat ──
        required = {
            "crsp_msi":   ("crsp_msi",        "CRSP Monthly Stock Index"),
            "crsp_msf":   ("crsp_msf",        "CRSP Monthly Stock File"),
            "mf_summary": ("crsp_mf_summary", "CRSP Mutual Fund Summary"),
            "compustat":  ("compustat_funda",  "Compustat Annual"),
            "ccm_link":   ("ccm_link",         "CRSP-Compustat Link"),
        }

        for attr, (cache_name, desc) in required.items():
            df = load_cache(cache_name)
            if df is None:
                raise FileNotFoundError(
                    f"{desc} not found in cache as '{cache_name}.parquet'. "
                    f"Ensure CACHE_DIR ({cfg.CACHE_DIR}) contains all parquets."
                )
            setattr(self, attr, df)
            logger.info(f"  {attr}: {df.shape}")

        # ── Fama-French factors ──
        ff5 = load_cache("ff5_monthly")
        mom = load_cache("mom_monthly")
        if ff5 is None:
            raise FileNotFoundError(
                f"FF5 factors not found in cache as 'ff5_monthly.parquet'. "
                f"Ensure CACHE_DIR ({cfg.CACHE_DIR}) contains all parquets."
            )
        if mom is None:
            raise FileNotFoundError(
                f"Momentum factor not found in cache as 'mom_monthly.parquet'. "
                f"Ensure CACHE_DIR ({cfg.CACHE_DIR}) contains all parquets."
            )
        # Replicate get_ff6_monthly merge logic
        if "date" in ff5.columns:
            ff5 = ff5.set_index("date")
        if "date" in mom.columns:
            mom = mom.set_index("date")
        self.ff_factors = ff5.join(mom, how="inner")
        logger.info(f"  ff_factors: {self.ff_factors.shape}")

        # ── FRED ──
        fred_all = load_cache("fred_all")
        if fred_all is not None:
            fred = fred_all.set_index("date") if "date" in fred_all.columns else fred_all
            self.fred_data = fred.resample("ME").last()
            logger.info(f"  fred_data: {self.fred_data.shape}")

            # Derive risk-free rate (TB3MS / 100 → monthly)
            if "tb3ms" in fred.columns:
                rf = fred["tb3ms"] / 100
                self.rf_rate = rf.resample("ME").last() / 12
                logger.info(f"  rf_rate: {len(self.rf_rate)} months")
            else:
                logger.warning("  rf_rate: tb3ms column not in fred_all cache")
        else:
            logger.warning("  fred_data: NOT AVAILABLE (fred_all not in cache)")

    def _download_free_data(self):
        """Download FRED series and momentum factor if not already cached."""
        from jappelli_experiments.data.fred_download import download_all_fred
        from jappelli_experiments.data.ff_download import get_momentum_monthly
        from jappelli_experiments.data.cache import load_cache

        fred_cached = load_cache("fred_all")
        if fred_cached is None:
            logger.info("  Downloading FRED series...")
            download_all_fred()
        else:
            logger.info("  FRED series already cached")

        mom_cached = load_cache("mom_monthly")
        if mom_cached is None:
            logger.info("  Downloading momentum factor...")
            get_momentum_monthly()
        else:
            logger.info("  Momentum factor already cached")

    # ── Phase 1: Compute intermediate variables ──────────────────

    def phase_1_compute_intermediates(self):
        """Classify funds, compute theta/Sharpe/flows, build panels."""
        if not self._phase_0_done:
            raise RuntimeError("Run phase_0_load_data() first.")

        logger.info("=" * 60)
        logger.info("PHASE 1: Computing intermediate variables")
        logger.info("=" * 60)

        from jappelli_experiments.data.panel_builder import (
            classify_static_funds, compute_theta_t, compute_fund_flows,
            compute_aggregate_flows, build_aggregate_monthly_panel,
            build_stock_month_panel,
        )
        from jappelli_experiments.shared.rolling_estimation import rolling_sharpe
        from jappelli_experiments.experiments.block_a.a2_aggregate_pressure import (
            compute_flow_pressure,
        )

        # 1. Fund classification
        logger.info("Step 1: Classifying funds...")
        self.fund_class = classify_static_funds(self.mf_summary)
        n_static = self.fund_class["is_static"].sum()
        n_dynamic = (~self.fund_class["is_static"]).sum()
        logger.info(f"  {n_static} static / {n_dynamic} dynamic funds")

        # 2. Theta_t
        logger.info("Step 2: Computing theta_t...")
        self.theta_t = compute_theta_t(self.mf_summary, self.fund_class)
        # Convert caldt index to month-end for alignment
        self.theta_t.index = pd.to_datetime(self.theta_t.index) + pd.offsets.MonthEnd(0)
        # Dedup in case multiple caldt map to same month-end
        self.theta_t = self.theta_t.groupby(self.theta_t.index).last()
        logger.info(f"  theta_t: {len(self.theta_t)} months, "
                    f"range [{self.theta_t.min():.2f}, {self.theta_t.max():.2f}]")

        # 3. Rolling Sharpe
        logger.info("Step 3: Computing rolling Sharpe ratio...")
        msi = self.crsp_msi.set_index("date").sort_index()
        msi.index = pd.to_datetime(msi.index) + pd.offsets.MonthEnd(0)
        vwretd = msi["vwretd"]
        rf_aligned = self.rf_rate.reindex(vwretd.index, method="nearest")
        self.sharpe_t = rolling_sharpe(vwretd, rf_aligned, window=12)
        self.sharpe_t.name = "sharpe_t"
        sharpe_valid = self.sharpe_t.dropna()
        logger.info(f"  sharpe_t: {len(sharpe_valid)} months, "
                    f"range [{sharpe_valid.min():.3f}, {sharpe_valid.max():.3f}]")

        # 4. Fund flows
        logger.info("Step 4: Computing fund flows...")
        self.fund_flows = compute_fund_flows(self.mf_summary)
        logger.info(f"  {len(self.fund_flows):,} fund-month flow observations")

        # 5. Aggregate flows
        logger.info("Step 5: Computing aggregate flows...")
        self.aggregate_flows = compute_aggregate_flows(
            self.fund_flows, self.fund_class
        )
        # Convert caldt index to month-end
        self.aggregate_flows.index = (
            pd.to_datetime(self.aggregate_flows.index) + pd.offsets.MonthEnd(0)
        )
        # Dedup
        self.aggregate_flows = self.aggregate_flows.groupby(
            self.aggregate_flows.index
        ).sum()
        logger.info(f"  aggregate_flows: {self.aggregate_flows.shape}, "
                    f"columns: {list(self.aggregate_flows.columns)}")

        # 6. Market value
        logger.info("Step 6: Extracting market value...")
        self.market_value = msi["totval"]
        self.market_value.name = "totval"

        # 7. Aggregate panel
        logger.info("Step 7: Building aggregate monthly panel...")
        self.aggregate_panel = build_aggregate_monthly_panel(
            crsp_msi=self.crsp_msi,
            fred_data=self.fred_data,
            ff_factors=self.ff_factors,
            theta=self.theta_t,
            sharpe=self.sharpe_t,
            force_rebuild=True,
        )
        logger.info(f"  aggregate_panel: {self.aggregate_panel.shape}")

        # 8. Flow pressure
        logger.info("Step 8: Computing flow pressure...")
        self.flow_pressure = compute_flow_pressure(
            self.aggregate_flows, self.market_value
        )

        # 9. Join flow pressure into aggregate panel
        flow_cols = ["static_flow_norm", "dynamic_flow_norm", "total_flow_norm"]
        available_flow_cols = [c for c in flow_cols if c in self.flow_pressure.columns]
        if available_flow_cols:
            self.aggregate_panel = self.aggregate_panel.join(
                self.flow_pressure[available_flow_cols], how="left"
            )
        logger.info(f"  aggregate_panel (with flows): {self.aggregate_panel.shape}")
        logger.info(f"  columns: {list(self.aggregate_panel.columns)}")

        # 10. Stock panel
        logger.info("Step 10: Building stock-month panel...")
        self.stock_panel = build_stock_month_panel(
            crsp_msf=self.crsp_msf,
            compustat=self.compustat,
            ccm_link=self.ccm_link,
            ff_factors=self.ff_factors,
            force_rebuild=True,
        )
        logger.info(f"  stock_panel: {self.stock_panel.shape}")

        self._phase_1_done = True
        logger.info("PHASE 1 COMPLETE")

    # ── Block A: Rational Bubble + Aggregate Pressure ────────────

    def run_block_a(self):
        """Run experiments A1 and A2."""
        self._require_phase_1()

        logger.info("=" * 60)
        logger.info("BLOCK A: Rational Bubble + Aggregate Pressure")
        logger.info("=" * 60)

        from jappelli_experiments.experiments.block_a.a1_rational_bubble import run_a1
        from jappelli_experiments.experiments.block_a.a2_aggregate_pressure import run_a2

        # A1
        logger.info("Running A1...")
        self.results["A1"] = run_a1(
            self.aggregate_panel, mf_summary=self.mf_summary
        )
        V_t = self.results["A1"].get("V_t")
        if V_t is not None:
            logger.info(f"  A1 V_t: {len(V_t.dropna())} observations saved")

        # A2
        logger.info("Running A2...")
        self.results["A2"] = run_a2(
            self.aggregate_panel,
            aggregate_flows=self.aggregate_flows,
            market_value=self.market_value,
            sp500_events=self.sp500_events,
        )

        logger.info("BLOCK A COMPLETE")

    # ── Block B: Cross-Sectional (year-by-year holdings) ─────────

    def run_block_b(self):
        """Run experiments B1 and B2 with memory-safe holdings processing."""
        self._require_phase_1()

        logger.info("=" * 60)
        logger.info("BLOCK B: Cross-Sectional Returns (Holdings-Based)")
        logger.info("=" * 60)

        from jappelli_experiments.experiments.block_b.b1_cross_sectional import (
            compute_static_ownership, run_b1,
        )
        from jappelli_experiments.experiments.block_b.b2_heterogeneous_effects import (
            run_b2,
        )

        # Compute static ownership year by year
        logger.info("Computing static ownership from holdings (year by year)...")
        import jappelli_experiments.config as cfg
        so_chunks = []
        holdings_files = sorted(cfg.CACHE_DIR.glob("crsp_mf_holdings_*.parquet"))

        if not holdings_files:
            logger.error("No holdings files found in cache. Cannot run Block B.")
            return

        for hf in holdings_files:
            logger.info(f"  Processing {hf.name}...")
            chunk = pd.read_parquet(hf)
            so_chunk = compute_static_ownership(
                chunk, self.fund_class, self.crsp_msf
            )
            so_chunks.append(so_chunk)
            del chunk
            gc.collect()

        self.so_it = pd.concat(so_chunks, ignore_index=True)
        del so_chunks
        gc.collect()
        logger.info(f"  Total SO observations: {len(self.so_it):,}")

        # Align to month-end and dedup
        self.so_it["date"] = (
            pd.to_datetime(self.so_it["date"]) + pd.offsets.MonthEnd(0)
        )
        self.so_it = self.so_it.drop_duplicates(
            subset=["permno", "date"], keep="last"
        )
        self.so_it = self.so_it.sort_values(["permno", "date"])

        # Merge SO into stock panel
        stock_panel_b = self.stock_panel.copy()
        stock_panel_b["date"] = (
            pd.to_datetime(stock_panel_b["date"]) + pd.offsets.MonthEnd(0)
        )
        so_cols = ["permno", "date", "static_ownership",
                   "dynamic_ownership", "total_mf_ownership"]
        available_so_cols = [c for c in so_cols if c in self.so_it.columns]
        stock_panel_b = pd.merge(
            stock_panel_b,
            self.so_it[available_so_cols],
            on=["permno", "date"],
            how="left",
        )

        # Forward-fill quarterly SO to monthly within each permno
        ownership_cols = [c for c in ["static_ownership", "dynamic_ownership",
                                       "total_mf_ownership"]
                         if c in stock_panel_b.columns]
        for col in ownership_cols:
            stock_panel_b[col] = stock_panel_b.groupby("permno")[col].ffill()

        so_coverage = stock_panel_b["static_ownership"].notna().mean()
        logger.info(f"  Stock panel with SO: {stock_panel_b.shape}, "
                    f"SO coverage: {so_coverage:.1%}")

        # B1 — pass panel that already has static_ownership
        logger.info("Running B1...")
        self.results["B1"] = run_b1(
            stock_panel_b,
            ff_factors=self.ff_factors,
            sp500_events=self.sp500_events,
        )

        # B2
        logger.info("Running B2...")
        self.results["B2"] = run_b2(
            stock_panel_b, ff_factors=self.ff_factors
        )

        logger.info("BLOCK B COMPLETE")

    # ── Block C: Bad Times + Causality ───────────────────────────

    def run_block_c(self):
        """Run experiments C1 and C2."""
        self._require_phase_1()

        logger.info("=" * 60)
        logger.info("BLOCK C: Bad-Times Amplification + Causality")
        logger.info("=" * 60)

        from jappelli_experiments.experiments.block_c.c1_bad_times import run_c1
        from jappelli_experiments.experiments.block_c.c2_inflow_return_causality import (
            run_c2,
        )

        # Get V_t from A1 results or cache
        V_t = None
        if "A1" in self.results and "V_t" in self.results["A1"]:
            V_t = self.results["A1"]["V_t"]

        # C1
        logger.info("Running C1...")
        self.results["C1"] = run_c1(self.aggregate_panel, V_t=V_t)

        # C2
        logger.info("Running C2...")
        self.results["C2"] = run_c2(self.aggregate_panel)

        logger.info("BLOCK C COMPLETE")

    # ── Block D: Flow Elasticity + State Dependence ──────────────

    def run_block_d(self):
        """Run experiments D1 and D2."""
        self._require_phase_1()

        logger.info("=" * 60)
        logger.info("BLOCK D: Flow Elasticity Decomposition")
        logger.info("=" * 60)

        from jappelli_experiments.experiments.block_d.d1_flow_elasticity import run_d1
        from jappelli_experiments.experiments.block_d.d2_state_dependent import run_d2

        # Try loading Haddad data
        haddad_data = None
        try:
            from jappelli_experiments.data.loaders import load_haddad_all_quarters
            haddad_data = load_haddad_all_quarters()
            logger.info(f"  Haddad data loaded: {haddad_data.shape}")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"  Haddad data not available: {e}")

        # D1
        logger.info("Running D1...")
        self.results["D1"] = run_d1(
            self.aggregate_panel,
            fund_flows=self.fund_flows,
            fund_class=self.fund_class,
            haddad_data=haddad_data,
        )

        # D2
        logger.info("Running D2...")
        self.results["D2"] = run_d2(self.aggregate_panel)

        logger.info("BLOCK D COMPLETE")

    # ── Block E: Extensions ──────────────────────────────────────

    def run_block_e(self):
        """Run experiments E1-E4."""
        self._require_phase_1()

        logger.info("=" * 60)
        logger.info("BLOCK E: Extensions")
        logger.info("=" * 60)

        from jappelli_experiments.experiments.block_e.e1_international import run_e1
        from jappelli_experiments.experiments.block_e.e2_stress_testing import run_e2
        from jappelli_experiments.experiments.block_e.e3_rebalancing import run_e3
        from jappelli_experiments.experiments.block_e.e4_factor_bubbles import run_e4

        # E1 — International
        logger.info("Running E1...")
        self.results["E1"] = run_e1(us_results=self.results.get("A2"))

        # E2 — Stress testing
        logger.info("Running E2...")
        V_t = None
        if "A1" in self.results and "V_t" in self.results["A1"]:
            V_t = self.results["A1"]["V_t"]

        daily_returns = None
        try:
            from jappelli_experiments.data.loaders import load_daily_returns
            daily_returns = load_daily_returns()
        except (FileNotFoundError, Exception):
            logger.warning("  Daily returns not available for E2")

        self.results["E2"] = run_e2(
            aggregate_panel=self.aggregate_panel,
            V_t=V_t,
            daily_returns=daily_returns,
        )

        # E3 — Rebalancing
        logger.info("Running E3...")
        holdings_e3 = self._load_holdings_subset(2020, 2023)
        self.results["E3"] = run_e3(
            self.aggregate_panel,
            holdings=holdings_e3,
            fund_class=self.fund_class,
            fund_flows=self.fund_flows,
        )
        del holdings_e3
        gc.collect()

        # E4 — Factor bubbles
        logger.info("Running E4...")
        self.results["E4"] = run_e4(
            stock_panel=self.stock_panel,
            ff_factors=self.ff_factors,
            so_it=self.so_it,
        )

        logger.info("BLOCK E COMPLETE")

    # ── Status ───────────────────────────────────────────────────

    def status(self):
        """Print current state of the orchestrator."""
        print("\n" + "=" * 60)
        print("ORCHESTRATOR STATUS")
        print("=" * 60)

        # Phase 0: Data
        print("\n-- Phase 0: Raw Data --")
        data_attrs = [
            ("crsp_msi", self.crsp_msi),
            ("crsp_msf", self.crsp_msf),
            ("mf_summary", self.mf_summary),
            ("compustat", self.compustat),
            ("ccm_link", self.ccm_link),
            ("ff_factors", self.ff_factors),
            ("fred_data", self.fred_data),
            ("rf_rate", self.rf_rate),
            ("sp500_events", self.sp500_events),
        ]
        for name, obj in data_attrs:
            if obj is None:
                print(f"  {name:20s}  NOT LOADED")
            elif hasattr(obj, "shape"):
                print(f"  {name:20s}  {str(obj.shape):>20s}")
            else:
                print(f"  {name:20s}  {len(obj):>10d} obs")

        # Phase 1: Intermediates
        print("\n-- Phase 1: Intermediates --")
        inter_attrs = [
            ("fund_class", self.fund_class),
            ("theta_t", self.theta_t),
            ("sharpe_t", self.sharpe_t),
            ("fund_flows", self.fund_flows),
            ("aggregate_flows", self.aggregate_flows),
            ("market_value", self.market_value),
            ("aggregate_panel", self.aggregate_panel),
            ("flow_pressure", self.flow_pressure),
            ("stock_panel", self.stock_panel),
            ("so_it", self.so_it),
        ]
        for name, obj in inter_attrs:
            if obj is None:
                print(f"  {name:20s}  NOT COMPUTED")
            elif hasattr(obj, "shape"):
                print(f"  {name:20s}  {str(obj.shape):>20s}")
            else:
                print(f"  {name:20s}  {len(obj):>10d} obs")

        # Results
        print("\n-- Experiment Results --")
        all_experiments = [
            "A1", "A2", "B1", "B2", "C1", "C2",
            "D1", "D2", "E1", "E2", "E3", "E4",
        ]
        for exp in all_experiments:
            if exp in self.results:
                r = self.results[exp]
                n_keys = len(r) if isinstance(r, dict) else 1
                print(f"  {exp:5s}  COMPLETE ({n_keys} result keys)")
            else:
                print(f"  {exp:5s}  NOT RUN")

        print("=" * 60 + "\n")

    # ── Helpers ──────────────────────────────────────────────────

    def _load_holdings_subset(self, start_year, end_year):
        """Load holdings for a range of years, handling H1/H2 naming."""
        import jappelli_experiments.config as cfg
        frames = []
        for f in sorted(cfg.CACHE_DIR.glob("crsp_mf_holdings_*.parquet")):
            stem = f.stem.replace("crsp_mf_holdings_", "")
            try:
                year = int(stem[:4])
            except ValueError:
                continue
            if start_year <= year <= end_year:
                frames.append(pd.read_parquet(f))
                logger.info(f"  Loaded {f.name}")

        if frames:
            return pd.concat(frames, ignore_index=True)
        logger.warning(f"No holdings files found for {start_year}-{end_year}")
        return pd.DataFrame()

    def _require_phase_1(self):
        """Guard: ensure phase 0 + 1 have been run."""
        if not self._phase_1_done:
            raise RuntimeError(
                "Run phase_0_load_data() and phase_1_compute_intermediates() first."
            )


# ── CLI entry point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Jappelli (2025) Experiment Orchestrator",
    )
    parser.add_argument(
        "--block", type=str, required=True,
        choices=["data", "a", "b", "c", "d", "e", "all", "status"],
        help="Which block to run (data=load+compute, a-e=experiment blocks, "
             "all=everything, status=print state)",
    )
    parser.add_argument(
        "--skip-downloads", action="store_true",
        help="Skip downloading free data (FRED, momentum)",
    )
    parser.add_argument(
        "--colab", action="store_true",
        help="Colab mode (cache-only, no WRDS connection)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Override cache directory (e.g., /content/drive/MyDrive/cache)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (e.g., /content/drive/MyDrive/jappelli_output)",
    )
    args = parser.parse_args()

    # Construct orchestrator first so output_dir/cache_dir patches take effect
    orch = ExperimentOrchestrator(
        skip_downloads=args.skip_downloads,
        colab_mode=args.colab,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    # Configure logging after orchestrator construction so LOG_DIR reflects patches
    import jappelli_experiments.config as cfg
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.LOG_DIR / "orchestrator.log", mode="a"),
        ],
    )

    if args.block == "status":
        orch.status()
        return

    # Always load data first
    orch.phase_0_load_data()

    if args.block == "data":
        orch.phase_1_compute_intermediates()
        orch.status()
        return

    # For experiment blocks, compute intermediates then run
    orch.phase_1_compute_intermediates()

    block_map = {
        "a": orch.run_block_a,
        "b": orch.run_block_b,
        "c": orch.run_block_c,
        "d": orch.run_block_d,
        "e": orch.run_block_e,
    }

    if args.block == "all":
        for block_fn in block_map.values():
            block_fn()
    else:
        block_map[args.block]()

    orch.status()


if __name__ == "__main__":
    main()
