from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "isFraud"
ID_COL = "TransactionID"



def basic_feature_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight transformations safe for baseline models."""
    out = df.copy()

    if "TransactionAmt" in out.columns:
        out["TransactionAmt_log1p"] = np.log1p(out["TransactionAmt"].fillna(0))

    if "TransactionDT" in out.columns:
        out["TransactionDay"] = (out["TransactionDT"] // (24 * 60 * 60)).astype("float")

    return out


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL]
    drop_cols: List[str] = [c for c in [TARGET_COL] if c in df.columns]
    x = df.drop(columns=drop_cols)
    return x, y


def infer_categorical_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(df[c].dtype) in ("object", "category")]
