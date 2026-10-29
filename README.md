# Capstone Fraud Detection

## Project Objective:
Real-time fraud detection and Agentic AI fraud triage:
Using IEEE-CIS fraud dataset, develop a machine learning real-time fraud detection model to assign fraud score to each transactions.
Further, develop an Agentic AI + RAG solution (Fraud Triage Agent) that supports the human fraud analyst in analyzing the fraud transaction.

Fraud triage for the IEEE-CIS fraud dataset (GNN + CatBoost model), in two parts:

```
Data/ ──► 1. Model pipeline (src/) ──► model_output/ ──► 2. Fraud triage agent (fraud_triage_agent/) ──► Streamlit UI
```

## 1. Model pipeline: [src/](src/README.md)
Ingests the Kaggle data, builds features and GNN embeddings, trains CatBoost, and writes the SHAP, validation and drift CSVs to `model_output/`.
```
python src/run_pipeline.py
```
Setup, configuration, steps and outputs: see [src/README.md](src/README.md).
(NOTE: The pipeline step is already run and the outputs saved in "model_output". Run only the #2 Fraud triage agent)

## 2. Fraud triage agent: [fraud_triage_agent/](fraud_triage_agent/README.md)
A multi-agent (LangGraph) assistant that summarises a blocked transaction for a fraud analyst, using three sub-agents (SHAP interpreter, historical look-up, model drift) and an orchestrator. It includes guardrails, an audit log, LangSmith tracing, offline evals and a Streamlit UI. It reads the pre-generated files in `model_output/`, so it runs without re-training.
```
pip install -r fraud_triage_agent/requirements.txt
streamlit run fraud_triage_agent/ui/streamlit_app.py
```
Copy `.env.example` to `.env` and add your keys first (never commit `.env`). Full instructions: see [fraud_triage_agent/README.md](fraud_triage_agent/README.md).

## Other files
`IEEE-CIS_Fraud_Model.ipynb` and `Agentic_AI.ipynb` hold the exploratory work.
