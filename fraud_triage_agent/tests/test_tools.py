import faiss
import numpy as np

from fraud_triage_agent.tools.history_tool import _normalize, _wilson
from fraud_triage_agent.tools.recommendation import recommend
from fraud_triage_agent.tools.drift_tool import _rank
from fraud_triage_agent.guardrails.output_guard import check_numeric_grounding, validate_interpretation
from fraud_triage_agent.schemas.outputs import TriageSummary


def test_recommendation_rules():
    """Recommendation follows the probability bands and the similar-case TP share."""
    assert recommend(0.96, 0, 0) == "High fraud probability, keep blocked"
    assert recommend(0.95, 5, 5) == "Likely fraud, review recommended"
    assert recommend(0.90, 8, 2) == "Likely fraud, review recommended"
    assert recommend(0.90, 3, 7) == "Likely false alarm, review recommended"
    assert recommend(0.90, 0, 0).startswith("No comparable history")
    assert recommend(None) is None


def test_faiss_matches_bruteforce_cosine():
    """FAISS inner-product search on normalised vectors must give the same neighbours as brute-force cosine."""
    rng = np.random.default_rng(0)
    data, query = rng.normal(size=(200, 16)), rng.normal(size=(1, 16))
    index = faiss.IndexFlatIP(16)
    index.add(_normalize(data))
    sim, idx = index.search(_normalize(query), 5)
    cos = (data @ query[0]) / (np.linalg.norm(data, axis=1) * np.linalg.norm(query))
    assert list(idx[0]) == list(np.argsort(-cos)[:5])
    assert np.allclose(1 - sim[0], 1 - np.sort(cos)[::-1][:5], atol=1e-5)


def test_wilson_bounds():
    """The interval for 8 of 10 should bracket 0.8 and stay inside (0, 1)."""
    lo, hi = _wilson(8, 10)
    assert 0 < lo < 0.8 < hi < 1


def test_drift_rank():
    """Drift statuses must order from stable to significant, ignoring case and the word "shift"."""
    assert _rank("moderate shift") == 1 and _rank("Significant shift") == 2 and _rank("stable") == 0
    assert _rank("Significant") > _rank("Moderate") > _rank("Stable")


def test_interpretation_rejects_digits():
    """A number in the LLM interpretation must make the output guardrail fail."""
    result = TriageSummary(
        transaction_id=1,
        summary="Deterministic summary.",
        interpretation="This looks risky in 80 percent of cases.",
    )
    try:
        check_numeric_grounding(result)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for numeric claim in interpretation.")


def test_interpretation_allows_feature_names_with_digits():
    """Digits inside feature names (Count_C1, GNN_Emb_8, V95) are not numeric claims."""
    validate_interpretation("Count_C1 and Days_Since_D2 raise risk; GNN_Emb_8 drifted, V95 did not.")


def test_interpretation_rejects_standalone_numbers():
    """Percentages, decimals and bare numbers in the interpretation must be rejected."""
    for text in ("About 80% were fraud.", "The top 5 drivers matter.", "Risk is 0.94."):
        try:
            validate_interpretation(text)
        except ValueError:
            continue
        raise AssertionError(f"Expected rejection: {text}")
