"""Run: streamlit run fraud_triage_agent/ui/streamlit_app.py"""
import math

import pandas as pd
import streamlit as st

from fraud_triage_agent.config import HIGH_RISK_CSV, ID_COLUMN, PROB_COLUMN
from fraud_triage_agent.agents.orchestrator import triage
from fraud_triage_agent.guardrails.input_guard import GuardrailError
from fraud_triage_agent.utils.audit import log_run

PAGE_SIZE = 50


@st.cache_data(show_spinner=False)
def load_candidates():
    df = pd.read_csv(HIGH_RISK_CSV, usecols=[ID_COLUMN, PROB_COLUMN])
    df = df.rename(columns={ID_COLUMN: "transaction_id", PROB_COLUMN: "fraud_proba"})
    return df.sort_values("fraud_proba", ascending=False).reset_index(drop=True)


with st.sidebar:
    st.header("Transactions")
    st.caption("Select a transaction_id cell and press Ctrl+C, then paste it into the text box.")
    try:
        cands = load_candidates()
    except Exception as e:
        cands = None
        st.error(f"Could not load transaction list: {e}")
    if cands is not None and not cands.empty:
        # Round bounds outward to the slider step so min/max rows are never excluded
        lo = math.floor(cands["fraud_proba"].min() * 1000) / 1000
        hi = math.ceil(cands["fraud_proba"].max() * 1000) / 1000
        rng = st.slider("fraud_proba range", lo, hi, (lo, hi), step=0.001, format="%.3f", key="proba_range")
        view = cands[(cands["fraud_proba"] >= rng[0]) & (cands["fraud_proba"] <= rng[1])]
        pages = max(1, -(-len(view) // PAGE_SIZE))
        st.caption(f"{len(view):,} transactions match")
        page = st.number_input(f"Page (of {pages})", 1, pages, 1, key="page") if pages > 1 else 1
        chunk = view.iloc[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
        st.dataframe(
            chunk, height=400, hide_index=True, use_container_width=True,
            key=f"tx_list_{rng[0]}_{rng[1]}_{page}",
        )
        

st.title("Fraud Triage Assistant")
st.caption("Decision support only - analyst confirms all actions.")

tid = st.text_input("Blocked transaction ID")
if st.button("Run triage") and tid:
    try:
        with st.spinner("Running agents..."):
            res = triage(tid)
        log_run(res)
        if res.recommendation:
            st.subheader("Recommendation")
            st.info(res.recommendation)
        st.subheader("Summary")
        st.write(res.summary)
        if res.interpretation:
            st.subheader("Evidence interpretation")
            st.write(res.interpretation)
        for c in res.caveats:
            st.warning(c)
        if res.history and res.history.matches:
            with st.expander(f"Similar historical cases ({len(res.history.matches)}) - verify in Validation_TP_FP_FN.csv"):
                st.dataframe([m.model_dump() for m in res.history.matches])
        with st.expander("Evidence (raw tool outputs)"):
            st.json(res.model_dump())
        st.caption(f"run_id: {res.run_id}")
    except GuardrailError as e:
        st.error(str(e))
