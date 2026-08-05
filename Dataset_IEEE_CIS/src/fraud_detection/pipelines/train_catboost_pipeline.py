from __future__ import annotations

from fraud_detection.data.loader import build_training_frame, load_raw_data
from fraud_detection.features.preprocessing import basic_feature_cleanup, split_xy
from fraud_detection.models.catboost_trainer import train_catboost



def run(sample: bool = False, nrows: int | None = None) -> None:
    bundle = load_raw_data(sample=sample, nrows=nrows)
    train_df = build_training_frame(bundle)
    train_df = basic_feature_cleanup(train_df)

    x, y = split_xy(train_df)
    result = train_catboost(x, y)
    print(f"Saved CatBoost model: {result.model_path}")
    print(f"Validation AUC: {result.val_auc:.6f}")
