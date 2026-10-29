"""Rule-based analyst recommendation (deterministic, no LLM)."""

HIGH_PROBA = 0.95  # above this, the model score alone justifies keeping the block
TP_RATE_CUTOFF = 0.50  # similar-case TP share at or above this points to real fraud


def recommend(fraud_probability: float | None, n_tp: int = 0, n_fp: int = 0) -> str | None:
    """Return a one-line action for the analyst, or None if the fraud probability is unavailable.

    - probability > 95%: keep blocked.
    - otherwise: use the TP share of similar historical cases (>= 50% likely fraud, else likely false alarm).
    - no TP/FP history to compare against: ask for a manual review instead of guessing.
    """
    if fraud_probability is None:
        return None
    if fraud_probability > HIGH_PROBA:
        return "High fraud probability, keep blocked"
    if n_tp + n_fp == 0:
        return "No comparable history found, manual review recommended"
    if n_tp / (n_tp + n_fp) >= TP_RATE_CUTOFF:
        return "Likely fraud, review recommended"
    return "Likely false alarm, review recommended"
