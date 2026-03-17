"""
Parquet-based caching for processed datasets.

Caches expensive computations (data cleaning, merges, variable construction)
to avoid recomputation across experiments.
"""
import hashlib
import logging
from pathlib import Path
from functools import wraps

import pandas as pd

from jappelli_experiments.config import CACHE_DIR

logger = logging.getLogger(__name__)


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key(name, **kwargs):
    """Generate a cache key from name and parameters."""
    param_str = str(sorted(kwargs.items()))
    h = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return f"{name}_{h}" if kwargs else name


def save_cache(df, name, **kwargs):
    """
    Save a DataFrame to Parquet cache.

    Parameters
    ----------
    df : DataFrame
        Data to cache.
    name : str
        Cache entry name.
    **kwargs
        Parameters that distinguish this cache entry.
    """
    _ensure_cache_dir()
    key = cache_key(name, **kwargs)
    path = CACHE_DIR / f"{key}.parquet"
    df.to_parquet(path, index=True)
    logger.info(f"Cached {key} ({len(df):,} rows) -> {path.name}")
    return path


def load_cache(name, **kwargs):
    """
    Load a DataFrame from Parquet cache.

    Returns
    -------
    DataFrame or None if cache miss.
    """
    key = cache_key(name, **kwargs)
    path = CACHE_DIR / f"{key}.parquet"
    if path.exists():
        logger.info(f"Cache hit: {key}")
        return pd.read_parquet(path)
    logger.info(f"Cache miss: {key}")
    return None


def clear_cache(name=None):
    """
    Clear cache entries.

    Parameters
    ----------
    name : str or None
        If provided, clear only entries matching this prefix.
        If None, clear all cached files.
    """
    _ensure_cache_dir()
    if name is None:
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        logger.info("Cleared all cache")
    else:
        for f in CACHE_DIR.glob(f"{name}*.parquet"):
            f.unlink()
            logger.info(f"Cleared cache: {f.name}")


def cached(name):
    """
    Decorator to cache function results as Parquet.

    Usage:
        @cached("aggregate_monthly_panel")
        def build_aggregate_monthly_panel():
            ...
            return df
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, force_rebuild=False, **kwargs):
            if not force_rebuild:
                result = load_cache(name)
                if result is not None:
                    return result
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                save_cache(result, name)
            return result
        return wrapper
    return decorator
