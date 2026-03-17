"""
Central configuration for Jappelli (2025) experiments.
Paths, constants, sample periods, and shared parameters.
"""
from pathlib import Path

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / "Data and Code"

# ── Output directories ──
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# ── Existing data paths ──
JIANG_DIR = DATA_ROOT / "Jiang" / "Empirical results"
JIANG_SEC6 = JIANG_DIR / "Section 6"
JIANG_APPENDIX = JIANG_DIR / "Appendices C E3"

OTHER_PAPER_DIR = DATA_ROOT / "Other paper" / "Replication Code-20221112" / "data"

HADDAD_DIR = DATA_ROOT / "Haddad 2"
HADDAD_ESTIMATION = HADDAD_DIR / "Estimation" / "output" / "estimation"
HADDAD_ANALYSIS = HADDAD_DIR / "Analysis" / "output"

BACKUS_DIR = DATA_ROOT / "Backus"
BACKUS_DATA = BACKUS_DIR / "data" / "public"
BACKUS_CODE = BACKUS_DIR / "code"

EXTRACTIONS_DIR = PROJECT_ROOT.parent / "extractions"

# ── Sample periods ──
SAMPLE_START = "1980-01-01"
SAMPLE_END = "2024-12-31"
MF_SAMPLE_START = "2004-01-01"   # CRSP MF summary availability
MF_SAMPLE_END = "2024-12-31"

# ── Static fund classification ──
STATIC_SD_THRESHOLD = 0.05       # SD(equity allocation) <= 5%
STATIC_SD_ALTERNATIVES = [0.03, 0.05, 0.07]  # Robustness thresholds
MIN_FUND_TNA = 5.0               # Minimum TNA in $M
ALLOC_BOUNDS = (0.75, 1.25)      # Valid allocation range

# ── Newey-West lags ──
NW_LAGS_QUARTERLY = 4            # As in Jappelli Table 1
NW_LAGS_MONTHLY = 6
NW_LAGS_DAILY = 10

# ── VAR settings ──
VAR_MAX_LAGS = 12
IRF_HORIZON = 24                 # Months

# ── Rolling window ──
ROLLING_WINDOW_MONTHS = 36
ROLLING_WINDOW_DAYS = 252        # ~1 trading year
ROLLING_BETA_WINDOW = 60         # 60-month rolling beta

# ── Fama-MacBeth settings ──
FM_MIN_OBS = 30                  # Minimum cross-section size

# ── IV diagnostics ──
IV_FIRST_STAGE_F_MIN = 10.0     # Stock-Yogo weak instruments

# ── Winsorization ──
WINSORIZE_PCTILE = (0.01, 0.99)

# ── FRED series ──
FRED_SERIES = {
    "DGS10": "10-Year Treasury Constant Maturity",
    "DGS2": "2-Year Treasury Constant Maturity",
    "TB3MS": "3-Month Treasury Bill",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "BAMLC0A0CM": "ICE BofA US Corporate Index OAS",
    "VIXCLS": "CBOE VIX",
    "CPIAUCSL": "CPI All Urban Consumers",
    "USREC": "NBER Recession Indicator",
}

FRED_URL_TEMPLATE = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id={series}&cosd={start}&coed={end}"
)

# ── Ken French library ──
FF_URL_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_DATASETS = {
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
}

# ── Shiller dataset ──
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

# ── International indices (Yahoo Finance) ──
INTERNATIONAL_INDICES = {
    "UK": "^FTSE",
    "Canada": "^GSPTSE",
    "Japan": "^N225",
    "Eurozone": "^STOXX50E",
    "US": "^GSPC",
}

# ── Existing data files (verified paths) ──
DATA_FILES = {
    "sp500": JIANG_SEC6 / "sp500.dta",
    "sp600": JIANG_SEC6 / "sp600.dta",
    "flow_iv": JIANG_SEC6 / "flow_iv.dta",
    "ret_daily": JIANG_APPENDIX / "ret_daily_seasonality.dta",
    "sp500_adddrop": OTHER_PAPER_DIR / "500diradddrop.dta",
    "russell": OTHER_PAPER_DIR / "russ_1990_2021_fix.dta",
    "daily_volume": OTHER_PAPER_DIR / "dailyvolume.dta",
}

# ── Random seed ──
RANDOM_SEED = 42
