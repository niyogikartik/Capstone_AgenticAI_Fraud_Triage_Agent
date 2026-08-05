from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_detection.agentic.risk_agent import explain_risk
from fraud_detection.serving.predict import load_model, predict_proba


st.set_page_config(page_title="Fraud Risk Workbench", layout="wide")
st.title("IEEE-CIS Fraud Detection Workbench")

model_path_default = Path("artifacts/models/catboost_fraud.cbm")
model_path_input = st.text_input("Model path", value=str(model_path_default))
uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None and st.button("Score Transactions"):
    model = load_model(Path(model_path_input))
    df = pd.read_csv(uploaded_file)
    scores = predict_proba(model, df)

    results = df.copy()
    results["fraud_probability"] = scores
    results["risk_band"] = results["fraud_probability"].apply(
        lambda s: explain_risk(float(s)).risk_band
    )

    st.subheader("Scored Results")
    st.dataframe(results.head(50))

    st.subheader("Risk Band Counts")
    st.bar_chart(results["risk_band"].value_counts())
