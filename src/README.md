NOTE: The pipieline section is already pre-run and the outputs are saved in "model_output" folder. The "fraud agent" can be run directly off of the files in "model_output" folder. If you wish to re-run the pipeline, you would need a GPU based kernel.

# Model Pipeline (`src/`)

Trains the fraud model end to end (data ingestion → feature engineering → GNN embeddings → CatBoost) and writes the CSVs that the [fraud triage agent](../fraud_triage_agent/README.md) reads.

## Setup

1. Create a Python environment and install the dependencies:
   ```
   pip install -r src/requirements.txt
   ```
2. Download the dataset and place the files under `Data/` in the repository root (not committed, due to size):
   - `train_identity.csv`, `train_transaction.csv`, `test_identity.csv`, `test_transaction.csv`
   - Source: https://www.kaggle.com/c/ieee-fraud-detection/data (or your shared Drive copy)

## Configure

Edit `src/config.yaml`:
- `data.*`: input CSV paths. Set `data.nrows` to a small number (e.g. 1000) for a smoke test, or `null` for the full dataset.
- `storage.output_dir`: where outputs are written (default `model_output`).
- `pipeline.run_*`: on/off flags for each step.
- `defaults.risk_threshold`: probability cutoff for "high risk" (default 0.86).

## Run

From the repository root, because the paths in the config are relative:
```
python src/run_pipeline.py
```

| Step | Script | Default | What it does |
|---|---|---|---|
| 01 | `01_data_ingestion.py` | on | Merge transaction and identity data; rename anonymised features |
| 02 | `02_feature_eng.py` | on | Velocity features; handle missing values |
| 03 | `03_final_dataset_for_training.py` | on | Drop sparse and correlated features; log amount; time features. Writes `final_dataset.csv` and `final_test_dataset.csv` |
| 04 | `04_EDA.py` | off | Correlation of numeric features with fraud |
| 05 | `05_gnn_embedder.py` | on | Train the GNN and produce 16-dimensional embeddings for train and test. Saves `gnn_model_state.pt` |
| 06 | `06_catboost_model.py` | on | Train and evaluate CatBoost; threshold metrics; SHAP; predict on the test set; run drift analysis |
| 07 | `07_shap_analysis.py` | off | Interactive only; not wired into the pipeline |

Steps build on each other, so disabling an early step means a later step needs its own inputs.

## Outputs (`model_output/`)

| File | Produced by | Used by the agent |
|---|---|---|
| `Test_Predicted_High_Risk_Transactions_Threshold_0.86.csv` | step 06 (test set, probability ≥ threshold): raw features, `fraud_proba`, `SHAP_*` columns | yes (SHAP interpreter) |
| `Validation_TP_FP_FN.csv` | step 06 (validation set): SHAP values with a `Category` of TP, FP or FN | yes (historical look-up) |
| `PSI_CSI_analysis.csv` | `model_drift.py` (columns `metric, name, value, status`) | yes (drift agent) |
| `catboost_model.cbm`, `gnn_model_state.pt` | steps 05, 06 | no (needed to re-score) |
| `test_predictions.csv` | step 06 | no |
| `feature_importance.csv`, `Threshold_metrics_v2.csv` | step 06 | no |
| `final_dataset.csv`, `final_test_dataset.csv`, `*_data_with_GNN_embeddings.csv` | steps 03, 05 | no |

The pre-generated outputs are already in `model_output/`, so the agent can be run without re-training.
