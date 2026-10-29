"""Central configuration. Override any value with environment variables."""
import os
from pathlib import Path

# Colab: mount Drive first (from google.colab import drive; drive.mount('/content/drive'))
DEFAULT_DATA_DIR = "/content/drive/Othercomputers/My Laptop/Capstone_Fraud_Detection/model_output"
DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", DEFAULT_DATA_DIR))

HIGH_RISK_CSV = DATA_DIR / "Test_Predicted_High_Risk_Transactions_Threshold_0.86.csv"
VALIDATION_CSV = DATA_DIR / "Validation_TP_FP_FN.csv"
DRIFT_CSV = DATA_DIR / "PSI_CSI_analysis.csv"

ID_COLUMN = os.getenv("FRAUD_ID_COLUMN", "TransactionID")
PROB_COLUMN = os.getenv("FRAUD_PROB_COLUMN", "fraud_proba")
LABEL_COLUMN = os.getenv("FRAUD_LABEL_COLUMN", "Category")  # TP / FP / FN
SHAP_PREFIX = os.getenv("FRAUD_SHAP_PREFIX", "SHAP_")  # prefix of SHAP columns

THRESHOLD = 0.86
TOP_K_FEATURES = 5
NEIGHBORS_K = 20
MAX_NEIGHBOR_DISTANCE = 0.5
PSI_MODERATE, PSI_SIGNIFICANT = 0.1, 0.25

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
AUDIT_LOG_DIR = Path(__file__).parent / "audit_logs"
PROMPT_VERSION = "v0.1"
