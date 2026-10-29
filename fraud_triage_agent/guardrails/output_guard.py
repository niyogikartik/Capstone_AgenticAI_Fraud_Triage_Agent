import re

from ..schemas.outputs import TriageSummary

BANNED_PHRASES = ["confirmed fraud", "guaranteed", "definitely fraud"]

# A standalone number: digits not attached to a word character, so feature
# names such as Count_C1, V95 or GNN_Emb_8 are not treated as numeric claims.
_STANDALONE_NUMBER = re.compile(r"(?<!\w)\d[\d,.]*%?")


def validate_interpretation(text: str) -> None:
    """Reject LLM text that makes banned claims (e.g. "confirmed fraud") or contains standalone numbers.

    Raises ValueError on failure.
    """
    low = text.lower()
    for p in BANNED_PHRASES:
        if p in low:
            raise ValueError(f"Unsupported claim in interpretation: '{p}'")
    # The orchestrator owns all quantitative statements; the LLM may only add
    # a qualitative interpretation without numbers.
    if _STANDALONE_NUMBER.search(text):
        raise ValueError("LLM interpretation must not introduce numeric claims.")


def check_numeric_grounding(result: TriageSummary) -> None:
    """Keep quantitative statements deterministic and reject unsupported claims."""
    low = result.summary.lower()
    for p in BANNED_PHRASES:
        if p in low:
            raise ValueError(f"Unsupported claim in summary: '{p}'")
    if result.interpretation:
        validate_interpretation(result.interpretation)
