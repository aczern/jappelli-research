"""
Data loaders for existing on-disk files (.dta, .parquet, .csv).

Reads from verified file locations and returns standardized DataFrames.
"""
import logging

import pandas as pd

from jappelli_experiments.config import (
    DATA_FILES, HADDAD_ESTIMATION, HADDAD_ANALYSIS,
    BACKUS_DATA, JIANG_SEC6, JIANG_APPENDIX, OTHER_PAPER_DIR,
)

logger = logging.getLogger(__name__)


def load_stata(name):
    """
    Load a Stata .dta file by name from DATA_FILES registry.

    Parameters
    ----------
    name : str
        Key from config.DATA_FILES (e.g., 'sp500', 'flow_iv').

    Returns
    -------
    DataFrame
    """
    path = DATA_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    logger.info(f"Loading {name} from {path}")
    return pd.read_stata(path)


def load_sp500():
    """Load S&P 500 stock-month panel from Jiang."""
    return load_stata("sp500")


def load_sp600():
    """Load S&P 600 stock-month panel from Jiang."""
    return load_stata("sp600")


def load_flow_iv():
    """Load flow instrumental variable data from Jiang."""
    return load_stata("flow_iv")


def load_sp500_adddrop():
    """Load S&P 500 additions/deletions for IV identification."""
    return load_stata("sp500_adddrop")


def load_russell():
    """Load Russell index reconstitution data."""
    return load_stata("russell")


def load_daily_returns():
    """Load daily returns with seasonality controls from Jiang."""
    return load_stata("ret_daily")


def load_daily_volume():
    """Load daily volume data."""
    return load_stata("daily_volume")


def load_haddad_elasticities(variant="Baseline_N"):
    """
    Load Haddad demand-system elasticity estimates.

    Parameters
    ----------
    variant : str
        Subdirectory name (e.g., 'Baseline_N', 'Instrument_Nl1y').

    Returns
    -------
    dict mapping quarter string -> DataFrame of elasticities
    """
    base = HADDAD_ESTIMATION / variant
    if not base.exists():
        raise FileNotFoundError(f"Haddad estimation directory not found: {base}")

    results = {}
    for f in sorted(base.glob("Ek_estimates_*.csv")):
        # Extract quarter from filename (e.g., 'Ek_estimates_IV_N_2001-03.csv')
        quarter = f.stem.split("_")[-1]
        results[quarter] = pd.read_csv(f)

    logger.info(f"Loaded {len(results)} quarters of Haddad {variant} estimates")
    return results


def load_ici_passive():
    """
    Load ICI passive investing share data.

    Searches multiple possible locations.

    Returns
    -------
    DataFrame
    """
    # Try primary location
    candidates = [
        HADDAD_ANALYSIS / "ICI_passive_investing.csv",
        HADDAD_ANALYSIS.parent / "ICI_passive_investing.csv",
    ]

    # Search recursively under Haddad 2
    from jappelli_experiments.config import HADDAD_DIR
    for f in HADDAD_DIR.rglob("ICI_passive_investing.csv"):
        candidates.append(f)

    for path in candidates:
        if path.exists():
            logger.info(f"Loading ICI passive investing from {path}")
            return pd.read_csv(path)

    logger.warning("ICI_passive_investing.csv not found. May need to generate from Haddad code.")
    return None


def load_backus_s34():
    """Load Backus scraped 13F data."""
    path = BACKUS_DATA / "out_scrape.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Backus data not found: {path}")
    logger.info(f"Loading Backus 13F data ({path.stat().st_size / 1e6:.0f} MB)")
    return pd.read_parquet(path)


def load_haddad_all_quarters(variant="Baseline_N"):
    """
    Load all quarterly Haddad elasticity estimates and stack into a single DataFrame.

    Returns
    -------
    DataFrame with added 'quarter' column.
    """
    quarterly = load_haddad_elasticities(variant)
    frames = []
    for q, df in quarterly.items():
        df = df.copy()
        df["quarter"] = q
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()
