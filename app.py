import streamlit as st
import pandas as pd
import asyncio
import matplotlib.pyplot as plt
from fraud_agent import build_vector_db, run_crewai_fraud_analysis

# Page configuration
st.set_page_config(page_title="Fraud Triage Agentic AI", layout="wide", page_icon="🛡️")

@st.cache_resource
def load_data():
    """
    Loads and caches the data and vector database to prevent reloading on every button click.
    """
    # Load dataset using relative paths for VS Code environment
    df_shap = pd.read_csv('Data/Test_Feature_SHAP_Threshold_0.86.csv')
    df_validation_refrence = pd.read_csv('Data/df_reference_sampled.csv')
    
    # Initialize FAISS Vector Database
    vector_db = build_vector_db(df_validation_refrence)
    return df_shap, df_validation_refrence, vector_db

st.title("🛡️ Agentic AI - Fraud Triage Application")
st.markdown("This Streamlit app interacts with the underlying **CrewAI** agents and **FAISS Vector DB** to provide Explainable AI summaries for high-risk credit card transactions.")

# Load Data
try:
    df_shap, df_validation_refrence, vector_db = load_data()
except Exception as e:
    st.error(f"Error loading CSV data. Make sure the files are present in the 'Data/' folder. Error: {e}")
    st.stop()

# ==========================================
# 1. TOP SECTION: Inputs
# ==========================================
st.markdown("---")
st.header("1. Analyst Input Query")

col1, col2 = st.columns([1, 2])

with col1:
    default_tx = df_shap['TransactionID'].iloc[0] if 'TransactionID' in df_shap.columns else "TEST_ID_1"
    transaction_id_input = st.text_input("Transaction ID", value=str(default_tx))
    
    # Try to convert to int if necessary (depends on dataset logic)
    try:
        transaction_id = int(transaction_id_input)
    except:
        transaction_id = transaction_id_input

with col2:
    question = st.text_input("Analyst Question", value="Why is the fraud probability so high?")

analyze_button = st.button("Run AI Fraud Triage Analysis", type="primary")

# ==========================================
# EXECUTION BLOCK
# ==========================================
if analyze_button:
    
    with st.spinner("Agents are analyzing SHAP values & searching Historical Vector DB... Please wait."):
        # Create a new event loop context for CrewAI async kickoff in Streamlit
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        top_5_series, final_report = loop.run_until_complete(
            run_crewai_fraud_analysis(transaction_id, df_shap, vector_db)
        )
    
    if top_5_series is None:
        st.error(final_report)
    else:
        # ==========================================
        # 2. MIDDLE SECTION: Agent Output
        # ==========================================
        st.markdown("---")
        st.header("2. Fraud Triage Agent Output")
        
        # Display the result from CrewAI
        st.success("Analysis complete.")
        st.markdown(f"**Analyst Query:** {question}")
        st.markdown(str(final_report))
        
        # ==========================================
        # 3. BOTTOM SECTION: Visual Plot
        # ==========================================
        st.markdown("---")
        st.header("3. Top 5 SHAP Value Contributors Visualization")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        top_5_series.sort_values(ascending=True).plot(kind='barh', color='salmon', ax=ax)
        
        ax.set_title(f'Top 5 SHAP Risk Features for Transaction {transaction_id}')
        ax.set_xlabel('SHAP Contribution Value')
        ax.set_ylabel('Variable Feature')
        plt.tight_layout()
        
        # Render the matplotlib plot via Streamlit
        st.pyplot(fig)
