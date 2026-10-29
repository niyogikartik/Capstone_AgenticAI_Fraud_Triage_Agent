import re

from ..tools import data_loader
from .. import config

_ID_RE = re.compile(r"^\d{5,10}$")


class GuardrailError(ValueError):
    """Raised when user input is rejected; its message is safe to show in the UI."""


def validate_transaction_id(raw: str) -> int:
    """Check the user's input is a 5-10 digit ID that exists in the blocked (high-risk) set.

    Returns the ID as an int, or raises GuardrailError.
    """
    raw = str(raw).strip()
    if not _ID_RE.match(raw):
        raise GuardrailError("Transaction ID must be 5-10 digits.")
    tid = int(raw)
    if tid not in set(data_loader.high_risk()[config.ID_COLUMN]):
        raise GuardrailError(f"Transaction {tid} is not in the blocked (high-risk) set.")
    return tid
