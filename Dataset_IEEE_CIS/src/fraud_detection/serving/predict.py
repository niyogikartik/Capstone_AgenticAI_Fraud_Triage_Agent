from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from fraud_detection.features.preprocessing import basic_feature_cleanup



def load_model(model_path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def predict_proba(model: CatBoostClassifier, df: pd.DataFrame) -> pd.Series:
    cleaned = basic_feature_cleanup(df)
    scores = model.predict_proba(cleaned)[:, 1]
    return pd.Series(scores, index=df.index, name="fraud_probability")
