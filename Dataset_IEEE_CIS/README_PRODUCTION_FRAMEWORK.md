# IEEE-CIS Fraud Platform: Modular + Production Framework

## 1) Target Architecture

- `src/fraud_detection/data`: ingestion, joins, schema contracts
- `src/fraud_detection/features`: EDA + feature engineering
- `src/fraud_detection/models`: model-specific training code (CatBoost, graph)
- `src/fraud_detection/pipelines`: orchestrated end-to-end workflows
- `src/fraud_detection/serving`: inference utilities
- `src/fraud_detection/agentic`: risk explanation and decision policies
- `app/streamlit_app.py`: analyst-facing interface
- `scripts/`: runnable entrypoints
- `tests/`: unit/integration tests
- `configs/`: experiment and environment-specific settings

## 2) Mapping Notebook Sections to Modules

Suggested migration approach from `IEEE-CIS_Fraud_Model.ipynb`:

1. Data loading + merges -> `data/loader.py`
2. EDA charts/tables -> `features/eda.py`
3. Feature transformations -> `features/preprocessing.py`
4. CatBoost training/evaluation -> `models/catboost_trainer.py`
5. Graph construction/training -> `models/graph_trainer.py`
6. Pipeline cells (run order) -> `pipelines/*.py`
7. UI cells -> `app/streamlit_app.py`
8. Explainability/rules/agent prompts -> `agentic/risk_agent.py`

## 3) Production Readiness Checklist

### Data Reliability
- Add strict schema validation (pandera or pydantic models)
- Version datasets and intermediate artifacts
- Log row counts, null rates, drift baselines per run

### Experiment Tracking
- Add MLflow/W&B for params, metrics, and artifacts
- Persist AUC, PR-AUC, KS, confusion matrix, threshold tables
- Store feature importance and SHAP summaries

### Reproducibility
- Pin dependencies in `requirements.txt` or `pyproject.toml`
- Make all randomness explicit (`random_state` everywhere)
- Save train/val splits and preprocessing state

### Testing
- Unit tests for feature functions and data joins
- Integration tests for `scripts/train_catboost.py`
- Contract tests for inference input schema

### Deployment
- Package inference into a FastAPI service or batch job
- Containerize with Docker
- CI/CD: lint, test, build image, deploy staging, then prod

### Monitoring
- Online: latency, error rate, throughput, confidence distribution
- Offline: daily drift, calibration decay, delayed-label AUC
- Alerting thresholds for model/data health regressions

### Security and Compliance
- Remove/obfuscate sensitive fields where needed
- Audit prediction and agent decision logs
- Use secrets manager for credentials and API keys

## 4) Suggested Operating Model

- Daily: score transactions + capture monitoring metrics
- Weekly: drift review and threshold recalibration
- Monthly: retrain candidate + shadow deployment + champion/challenger evaluation

## 5) Next Increment Plan

1. Migrate first 20-30 notebook cells into the created modules.
2. Add MLflow tracking wrapper around pipeline runs.
3. Add FastAPI endpoint (`/predict`) and Dockerfile.
4. Set up GitHub Actions for tests and quality checks.
