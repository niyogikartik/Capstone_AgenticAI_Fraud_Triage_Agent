"""Deterministic top-N SHAP lookup (no LLM)."""
from .. import config
from ..schemas.outputs import FeatureContribution, ShapReport
from . import data_loader


def get_top_shap_features(transaction_id: int, n: int = config.TOP_K_FEATURES) -> ShapReport:
    """Return the fraud probability and the n features with the largest absolute SHAP value
    for one blocked transaction, each with its direction (raises or lowers risk)."""
    df = data_loader.high_risk()
    rows = df[df[config.ID_COLUMN] == transaction_id]
    if rows.empty:
        raise KeyError(f"Transaction {transaction_id} not in high-risk set")
    row = rows.iloc[0]
    cols = data_loader.shap_columns(df)
    top = row[cols].abs().sort_values(ascending=False).head(n).index
    feats = []
    for c in top:
        name = c[len(config.SHAP_PREFIX):]
        val = float(row[c])
        feats.append(FeatureContribution(
            feature=name,
            shap_value=val,
            feature_value=row.get(name),
            direction="increases_risk" if val > 0 else "decreases_risk",
        ))
    return ShapReport(transaction_id=transaction_id,
                      fraud_probability=float(row[config.PROB_COLUMN]),
                      top_features=feats)
