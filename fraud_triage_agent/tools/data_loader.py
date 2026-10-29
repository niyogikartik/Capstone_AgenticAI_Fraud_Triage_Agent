"""Cached CSV loaders. All tools read data through here."""
from functools import lru_cache

import pandas as pd

from .. import config


@lru_cache(maxsize=1)
def high_risk() -> pd.DataFrame:
    """Load (once) the blocked test transactions: raw features, fraud probability and SHAP values."""
    return pd.read_csv(config.HIGH_RISK_CSV)


@lru_cache(maxsize=1)
def validation() -> pd.DataFrame:
    """Load (once) the historical validation SHAP values labelled TP / FP / FN."""
    return pd.read_csv(config.VALIDATION_CSV)


@lru_cache(maxsize=1)
def drift() -> pd.DataFrame:
    """Load (once) the PSI/CSI model drift table (metric, name, value, status)."""
    return pd.read_csv(config.DRIFT_CSV)


def shap_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of the SHAP columns in a dataframe (those starting with the SHAP prefix)."""
    return [c for c in df.columns if c.startswith(config.SHAP_PREFIX)]
