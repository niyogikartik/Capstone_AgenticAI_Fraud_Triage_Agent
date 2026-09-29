import os
import faiss
import numpy as np
import pandas as pd
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

# Load local environment variables for the API key
load_dotenv('.env')

def build_vector_db(df_validation):
    """
    Builds a FAISS Vector Database for Nearest Neighbor search.
    Encodes all SHAP feature columns into a high-dimensional vector space.
    """
    # Isolate all SHAP columns and sort alphabetically for consistency
    shap_cols = [c for c in df_validation.columns if str(c).startswith('Shap_')]
    shap_cols.sort()
    
    # Extract the numeric values for the FAISS index
    # (FAISS requires float32 contiguous arrays)
    vectors = np.ascontiguousarray(df_validation[shap_cols].fillna(0).values.astype('float32'))
    
    # Build FAISS Index (L2 = Euclidean Distance)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    
    # Keep track of the categories assigned to these vectors
    categories = df_validation['Category'].values if 'Category' in df_validation.columns else np.array(['Unknown'] * len(df_validation))
    
    return index, shap_cols, categories

async def run_crewai_fraud_analysis(transaction_id, df_shap, vector_db):
    """
    CrewAI Framework orchestrating the SHAP Interpreter and Fraud Triage agents.
    Returns the top 5 SHAP series (for plotting) and the final report from the agents.
    """
    index, shap_cols, categories = vector_db
    
    # 1. Fetch Transaction Data
    tx_data = df_shap[df_shap['TransactionID'] == transaction_id]
    if tx_data.empty: 
        return None, f"Transaction {transaction_id} not found."
    
    probability = tx_data['Predicted_Fraud_Probability'].values[0] if 'Predicted_Fraud_Probability' in tx_data.columns else "N/A"
    
    # For Interpret Agent: Get the top 5 highest contributing SHAP values
    top_5_series = tx_data[[c for c in shap_cols if c in tx_data.columns]].iloc[0].sort_values(ascending=False).head(5)
    tx_features_top_5 = top_5_series.to_dict()
    
    # 2. Vector DB Approximate Nearest Neighbor Search
    # Align the test transaction completely to the Vector DB's expected columns using **ALL SHAP variables**
    tx_vector_df = tx_data.reindex(columns=shap_cols, fill_value=0)
    tx_vector = np.ascontiguousarray(tx_vector_df.values.astype('float32'))
    
    # Perform Search (Top 5 matches via Euclidean distance using the entire multi-dimensional SHAP space)
    distances, indices = index.search(tx_vector, 5)
    
    # Map matched indices back to their target categories
    closest_categories = categories[indices[0]]
    most_likely_category = pd.Series(closest_categories).mode()[0]
    
    reference_data_sample_str = f"Top 5 ANN Matching Categories from Vector DB (Based on ALL SHAP features): {closest_categories.tolist()}\nEuclidean Distances: {distances[0].tolist()}\n\n--> ANN Selected Category: {most_likely_category}"
    
    # ==========================================
    # Define Agents
    # ==========================================
    shap_interpreter = Agent(
        role='SHAP Interpreter Analyst',
        goal='Interpret SHAP values to explain the key factors driving high fraud probabilities.',
        backstory='You are an expert Data Scientist specializing in model explainability and fraud detection. You break down complex feature attributes into simple business risk factors.',
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    fraud_triage_specialist = Agent(
        role='Fraud Triage Specialist',
        goal='Provide an actionable recommendation based on SHAP explanations and Vector DB nearest-neighbor search.',
        backstory='You are a Senior Fraud Analyst. You evaluate explainability metrics and cross-reference them with historical patterns retrieved from Vector DBs to make final block/allow decisions.',
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )
    
    # ==========================================
    # Define Tasks
    # ==========================================
    interpret_task = Task(
        description=f'''
        Analyze Transaction {transaction_id} with Fraud Probability: {probability}.
        The top driving SHAP features and their values are: {tx_features_top_5}.
        
        Provide a concise summary explaining to a human analyst why this transaction was flagged.
        ''',
        expected_output='A brief paragraph explaining the main reasons the model predicted high fraud risk.',
        agent=shap_interpreter
    )

    triage_task = Task(
        description=f'''
        Review the SHAP interpretation for Transaction {transaction_id} (Probability: {probability}).
        Here is the historical pattern matching result from the FAISS Vector Database search using Euclidean distance on ALL SHAP vectors: 
        {reference_data_sample_str}

        Combine the SHAP context with the Vector DB match to decide the best action. 
        If the likely category is FP (False Positive), lean towards recommendations like "Request 3D authentication" or "Allow with monitoring". 
        If TP (True Positive), lean towards "Keep blocked".
        If FN (False Negative), lean towards "Manual Review. Even though the probability is low, it might be a fraud transaction".
        ''',
        expected_output=f'''
        Return the output STRICTLY in the following format:
        1. Recommendation: [e.g. Request 3D authentication / Manual review / keep blocked]
        2. Fraud probability: [{probability}]
        3. Justification: [Your reasoning factoring in the matching {most_likely_category} category...]
        ''',
        agent=fraud_triage_specialist
    )

    # ==========================================
    # Assemble & Run Crew
    # ==========================================
    crew = Crew(
        agents=[shap_interpreter, fraud_triage_specialist],
        tasks=[interpret_task, triage_task],
        verbose=True,
        process=Process.sequential
    )

    result = await crew.kickoff_async()
    return top_5_series, result
