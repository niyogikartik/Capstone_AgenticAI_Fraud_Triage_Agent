from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fraud_detection.features.preprocessing import infer_categorical_features
from fraud_detection.settings import get_project_paths
from fraud_detection.utils.io import ensure_dir


@dataclass
class CatBoostTrainResult:
    model_path: Path
    val_auc: float



def train_catboost(x, y, random_state: int = 42) -> CatBoostTrainResult:
    paths = get_project_paths()
    out_dir = ensure_dir(paths.artifacts_root / "models")
    model_path = out_dir / "catboost_fraud.cbm"

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    cat_features = infer_categorical_features(x_train)

    train_pool = Pool(x_train, y_train, cat_features=cat_features)
    val_pool = Pool(x_val, y_val, cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.05,
        depth=8,
        iterations=1000,
        random_seed=random_state,
        verbose=200,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    model.save_model(str(model_path))

    return CatBoostTrainResult(model_path=model_path, val_auc=auc)
