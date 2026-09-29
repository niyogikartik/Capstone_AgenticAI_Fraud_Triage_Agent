# Capstone Fraud Detection

IEEE-CIS fraud detection pipeline: data ingestion → feature engineering → GNN embeddings → CatBoost model, plus an agentic app for inference.

## Setup

1. Create a Python environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your own API key(s):
   ```
   cp .env.example .env
   ```
3. Download the dataset and place the files under `Data/` (not committed to this repo due to file size):
   - Google Drive: `<add-your-shared-drive-link-here>`
   - Or the original competition data: https://www.kaggle.com/c/ieee-fraud-detection/data

## Running the training pipeline

Edit `src/config.yaml` to toggle which steps run (`pipeline.run_*` flags) and set `data.nrows` to a small number (e.g. 1000) for a quick smoke test, or `null` for the full dataset. Then run:
```
python src/run_pipeline.py
```

## Running the app

```
python app.py
```
