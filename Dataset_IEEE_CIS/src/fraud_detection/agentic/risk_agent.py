from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskAgentDecision:
    score: float
    risk_band: str
    rationale: str



def explain_risk(score: float, top_features: list[str] | None = None) -> RiskAgentDecision:
    """Rule-based placeholder for an agentic decision layer.

    Replace this with an LLM or orchestration framework later (LangGraph/AutoGen/CrewAI).
    """
    if score >= 0.9:
        band = "critical"
    elif score >= 0.7:
        band = "high"
    elif score >= 0.4:
        band = "medium"
    else:
        band = "low"

    feature_text = ", ".join(top_features or [])
    rationale = f"Risk classified as {band}. Key drivers: {feature_text or 'not provided'}."
    return RiskAgentDecision(score=score, risk_band=band, rationale=rationale)


def to_dict(decision: RiskAgentDecision) -> dict[str, Any]:
    return {
        "score": decision.score,
        "risk_band": decision.risk_band,
        "rationale": decision.rationale,
    }
