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
    print(f"Source Directory: {SRC_DIR}")
    with open(SRC_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    flags = config.get("pipeline", {})

    def enabled(script):
        return flags.get(f"run_{script}", False)

    # Setting up the train dataset, train GNN embeddings and training the CatBoost model
    df = None
    train_dataset_columns = []

    if enabled("01_data_ingestion.py"):
        print("Running 01_data_ingestion.py")
        ingestion = _load("01_data_ingestion.py")
        df = ingestion.load_and_merge_train_data()
        df = ingestion.rename_anonymized_features(df)
        print("Completed 01_data_ingestion.py")

    if enabled("02_feature_eng.py"):
        print("Running 02_feature_eng.py")
        feature_eng = _load("02_feature_eng.py")
        df = feature_eng.create_velocity_features(df)
        df = feature_eng.handle_missing_values(df)
        print("Completed 02_feature_eng.py")

    if enabled("03_final_dataset_for_training.py"):
        print("Running 03_final_dataset_for_training.py")
        final_dataset = _load("03_final_dataset_for_training.py")
        df, sparse_col_to_drop = final_dataset.identify_and_drop_sparse_features(df)
        df, high_corr_col_to_drop = final_dataset.remove_highly_correlated_features(df)
        df = final_dataset.log_transform_amount(df)
        df = final_dataset.extract_temporal_and_cyclical_features(df)
        train_dataset_columns = list(df.columns)
        print(f"Captured {len(train_dataset_columns)} finalized training columns.")
        output_dir = Path(config.get("storage", {}).get("output_dir", "model_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "final_dataset.csv.gz", index=False, compression='gzip')
        print("Completed 03_final_dataset_for_training.py")

    if enabled("04_EDA.py"):
        print("Running 04_EDA.py")
        eda = _load("04_EDA.py")
        eda.correlation_of_num_features_with_fraud(df)
        print("Completed 04_EDA.py")

    enriched_features = None
    if enabled("05_gnn_embedder.py"):
        print("Running 05_gnn_embedder.py")
        gnn_embedder = _load("05_gnn_embedder.py")
        graph_data, txn_features = gnn_embedder.generate_graph_data(df)
        output_dir = Path(config.get("storage", {}).get("output_dir", "model_output"))
        enriched_features = gnn_embedder.train_gnn_model(
            graph_data, txn_features, output_dir=output_dir
        )
        print("Completed 05_gnn_embedder.py")

    if enabled("06_catboost_model.py"):
        print("Running 06_catboost_model.py")
        catboost_model = _load("06_catboost_model.py")
        txn_labels = df['isFraud']
        clf_tuned, train_idx, val_idx, y_train, y_val = catboost_model.train_catboost_model(
            enriched_features, txn_labels
        )
        X_val_refined = enriched_features[val_idx]
        y_val, y_pred_proba_tuned, y_pred_tuned, _ = catboost_model.evaluate_tuned_catboost_model(
            clf_tuned, X_val_refined, y_val
        )

        output_dir = Path(config.get("storage", {}).get("output_dir", "model_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        clf_tuned.save_model(str(output_dir / "catboost_model.cbm"))
        print(f"Saved CatBoost model to {output_dir / 'catboost_model.cbm'}")
        catboost_model.get_feature_importance(
            clf_tuned,
            list(_load("05_gnn_embedder.py").TXN_FEATURE_COLS),
            output_dir=output_dir,
        )
        catboost_model.calcluate_tuned_threshold(
            y_pred_proba_tuned, y_val, output_dir=output_dir
        )
        catboost_model.calculate_SHAP_values(
            clf_tuned,
            X_val_refined,
            y_val,
            y_pred_tuned,
            threshold=0.86,
            feature_names=list(_load("05_gnn_embedder.py").TXN_FEATURE_COLS)
            + [f"GNN_Emb_{i}" for i in range(16)],
            output_dir=output_dir,
        )
        print("Completed 06_catboost_model.py")

    if enabled("07_shap_analysis.py"):
        print("Skipping 07_shap_analysis.py: run interactively, not wired into the automated pipeline yet.")


    #################################################
    # Setting up test dataset
    # Setting up the train dataset, train GNN embeddings and training the CatBoost model
    print("Setting up test dataset")
    df_test = None

    if enabled("01_data_ingestion.py"):
        print("Running 01_data_ingestion.py")
        ingestion = _load("01_data_ingestion.py")
        df_test = ingestion.load_and_merge_test_data()
        df_test = ingestion.rename_anonymized_features(df_test)
        print("Completed 01_data_ingestion.py")

    if enabled("02_feature_eng.py"):
        print("Running 02_feature_eng.py")
        feature_eng = _load("02_feature_eng.py")
        df_test = feature_eng.create_velocity_features(df_test)
        df_test = feature_eng.handle_missing_values(df_test)
        print("Completed 02_feature_eng.py")

    if enabled("03_final_dataset_for_training.py"):
        print("Running 03_final_dataset_for_training.py")
        final_dataset = _load("03_final_dataset_for_training.py")
        
        df_test = final_dataset.log_transform_amount(df_test)
        df_test = final_dataset.extract_temporal_and_cyclical_features(df_test)
        test_feature_columns = [
            column for column in train_dataset_columns if column != "isFraud"
        ]
        df_test_final = df_test[test_feature_columns]
        print(f"Final test dataset shape: {df_test_final.shape}")
        output_dir = Path(config.get("storage", {}).get("output_dir", "model_output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        df_test_final.to_csv(output_dir / "final_test_dataset.csv.gz", index=False, compression='gzip')
        print("Completed 03_final_dataset_for_training.py")

    if enabled("05_gnn_embedder.py"):
        print("Running 05_gnn_embedder.py on test data")
        gnn_embedder = _load("05_gnn_embedder.py")
        output_dir = Path(config.get("storage", {}).get("output_dir", "model_output"))
        graph_data_test, txn_features_test = gnn_embedder.generate_graph_data(df_test)
        enriched_features_test = gnn_embedder.generate_gnn_embeddings(
            graph_data_test,
            txn_features_test,
            model_path=output_dir / "gnn_model_state.pt",
            output_dir=output_dir,
        )
        print("Completed 05_gnn_embedder.py on test data")

        if enabled("06_catboost_model.py"):
            import pandas as pd
            from catboost import CatBoostClassifier

            print("Running CatBoost prediction on test data")
            clf = CatBoostClassifier()
            clf.load_model(str(output_dir / "catboost_model.cbm"))
            fraud_proba = clf.predict_proba(enriched_features_test)[:, 1]
            submission = pd.DataFrame(
                {"TransactionID": df_test["TransactionID"].values, "isFraud": fraud_proba}
            )
            submission.to_csv(output_dir / "test_predictions.csv", index=False)
            print(f"Saved test predictions to {output_dir / 'test_predictions.csv'}")

            import numpy as np
            import shap

            risk_threshold = config.get("defaults", {}).get("risk_threshold", 0.86)
            high_risk = fraud_proba >= risk_threshold
            feature_names = list(gnn_embedder.TXN_FEATURE_COLS) + [
                f"GNN_Emb_{i}" for i in range(16)
            ]
            print(f"Max test fraud_proba: {fraud_proba.max():.4f}; rows >= {risk_threshold}: {high_risk.sum()}")
            if high_risk.any():
                shap_high_risk = shap.TreeExplainer(clf)(enriched_features_test[high_risk]).values
            else:
                shap_high_risk = np.empty((0, len(feature_names)))
            high_risk_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "TransactionID": df_test["TransactionID"].values[high_risk],
                            "fraud_proba": fraud_proba[high_risk],
                        }
                    ),
                    pd.DataFrame(enriched_features_test[high_risk], columns=feature_names),
                    pd.DataFrame(
                        shap_high_risk, columns=[f"SHAP_{n}" for n in feature_names]
                    ),
                ],
                axis=1,
            )
            high_risk_path = (
                output_dir / f"Test_Predicted_High_Risk_Transactions_Threshold_{risk_threshold}.csv"
            )
            high_risk_df.to_csv(high_risk_path, index=False)
            print(f"Saved {len(high_risk_df)} high-risk transactions to {high_risk_path}")

            if enriched_features is not None:
                print("Running model_drift.py")
                model_drift = _load("model_drift.py")
                model_drift.run_drift_analysis(
                    train_proba=clf.predict_proba(enriched_features)[:, 1],
                    test_proba=fraud_proba,
                    train_features=enriched_features,
                    test_features=enriched_features_test,
                    feature_names=feature_names,
                    output_dir=output_dir,
                )
                print("Completed model_drift.py")
            else:
                print("Skipping model_drift.py: train embeddings unavailable.")


if __name__ == "__main__":
    main()
