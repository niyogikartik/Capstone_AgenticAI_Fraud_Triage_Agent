"""PSI/CSI drift lookup. File columns: metric (PSI/CSI), name, value, status."""
from ..schemas.outputs import DriftFeature, DriftReport
from . import data_loader

_RANK = {"stable": 0, "moderate": 1, "significant": 2}


def _rank(status: str) -> int:
    """Turn a drift status label into a number so statuses can be compared (higher = worse).

    Matches on keywords because the drift file uses "moderate shift" / "significant shift".
    """
    s = status.lower()
    for key, rank in (("significant", 2), ("moderate", 1), ("stable", 0)):
        if key in s:
            return rank
    return 1


def get_drift_report(top_features: list[str] | None = None, n: int = 10) -> DriftReport:
    """Summarize model drift: the overall status, the n most drifted features, and the drift
    row for every feature in top_features (flagged so the summary can call out relevant drift)."""
    df = data_loader.drift().sort_values("value", ascending=False)
    top = set(top_features or [])
    # worst n drifters, plus the drift row of every feature driving this transaction
    sel = df[df.index.isin(df.head(n).index) | df["name"].isin(top)]
    rows = [DriftFeature(feature=str(r["name"]), psi=float(r["value"]),
                         status=str(r["status"]).lower(),
                         in_transaction_top_features=str(r["name"]) in top)
            for _, r in sel.iterrows()]
    worst = max((str(s).lower() for s in df["status"]), key=_rank, default="stable")
    return DriftReport(overall_status=worst, drifting_features=rows)
