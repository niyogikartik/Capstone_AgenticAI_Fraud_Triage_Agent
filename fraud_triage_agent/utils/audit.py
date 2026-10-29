import json
import time

from .. import config
from ..schemas.outputs import TriageSummary


def log_run(result: TriageSummary, user: str = "analyst") -> None:
    """Append one triage run to audit_logs/audit.jsonl with timestamp, user, model and prompt version,
    so every answer can be traced back later."""
    config.AUDIT_LOG_DIR.mkdir(exist_ok=True)
    rec = {"ts": time.time(), "user": user, "prompt_version": config.PROMPT_VERSION,
           "model": config.LLM_MODEL, "result": result.model_dump()}
    with open(config.AUDIT_LOG_DIR / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
