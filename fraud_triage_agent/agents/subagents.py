"""Sub-agents. Each wraps one deterministic tool; add LLM narration later."""
from langsmith import traceable

from ..schemas.outputs import DriftReport, HistoricalReport, ShapReport
from ..tools.drift_tool import get_drift_report
from ..tools.history_tool import find_similar_history
from ..tools.shap_tool import get_top_shap_features


@traceable(name="shap_interpreter_agent", tags=["subagent"])
def shap_interpreter_agent(transaction_id: int) -> ShapReport:
    """Sub-agent 1: explain WHY the model flagged this transaction (its top 5 SHAP features)."""
    return get_top_shap_features(transaction_id)


@traceable(name="historical_lookup_agent", tags=["subagent"])
def historical_lookup_agent(transaction_id: int) -> HistoricalReport:
    """Sub-agent 2: find past validation cases with similar SHAP patterns and report how many were TP vs FP."""
    return find_similar_history(transaction_id)


@traceable(name="model_drift_agent", tags=["subagent"])
def model_drift_agent(top_features: list[str] | None = None) -> DriftReport:
    """Sub-agent 3: report model drift (PSI/CSI), highlighting any of the given top features."""
    return get_drift_report(top_features)
