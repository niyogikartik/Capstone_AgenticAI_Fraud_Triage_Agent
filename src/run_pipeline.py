"""Trains the fraud model end-to-end using the functions defined in src/*.py.

Gated by True/False flags under `pipeline` in config.yaml. Steps build on each
other (e.g. GNN training needs the feature-engineered dataframe from steps
01-03), so disabling an early step means later steps must supply their own
inputs.
"""
import importlib
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))


def _load(module_file):
    """Imports a src/NN_name.py module by its filename (without extension)."""
    return importlib.import_module(module_file[:-3])


def main():
    with open(SRC_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    flags = config.get("pipeline", {})

    def enabled(script):
        return flags.get(f"run_{script}", False)

    df = None

    if enabled("01_data_ingestion.py"):
        ingestion = _load("01_data_ingestion.py")
        df = ingestion.load_and_merge_train_data()
        df = ingestion.rename_anonymized_features(df)

    if enabled("02_feature_eng.py"):
        feature_eng = _load("02_feature_eng.py")
        df = feature_eng.create_velocity_features(df)
        df = feature_eng.handle_missing_values(df)

    if enabled("03_final_dataset_for_training.py"):
        final_dataset = _load("03_final_dataset_for_training.py")
        df, _ = final_dataset.identify_and_drop_sparse_features(df)
        df, _ = final_dataset.remove_highly_correlated_features(df)
        df = final_dataset.log_transform_amount(df)
        df = final_dataset.extract_temporal_and_cyclical_features(df)

    if enabled("04_EDA.py"):
        eda = _load("04_EDA.py")
        eda.correlation_of_num_features_with_fraud(df)

    enriched_features = None
    if enabled("05_gnn_embedder.py"):
        gnn_embedder = _load("05_gnn_embedder.py")
        graph_data, txn_features = gnn_embedder.generate_graph_data(df)
        enriched_features = gnn_embedder.train_gnn_model(graph_data, txn_features)

    if enabled("06_catboost_model.py"):
        catboost_model = _load("06_catboost_model.py")
        txn_labels = df['isFraud']
        clf_tuned, train_idx, val_idx, y_train, y_val = catboost_model.train_catboost_model(
            enriched_features, txn_labels
        )
        X_val_refined = enriched_features[val_idx]
        catboost_model.evaluate_tuned_catboost_model(clf_tuned, X_val_refined, y_val)

        output_dir = Path(config.get("storage", {}).get("output_dir", "model_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        clf_tuned.save_model(str(output_dir / "catboost_model.cbm"))
        print(f"Saved CatBoost model to {output_dir / 'catboost_model.cbm'}")

    if enabled("07_shap_analysis.py"):
        print("Skipping 07_shap_analysis.py: run interactively, not wired into the automated pipeline yet.")


if __name__ == "__main__":
    main()
