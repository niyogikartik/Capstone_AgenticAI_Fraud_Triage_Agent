"""Model drift analysis: PSI of fraud probabilities and CSI of model input features."""
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6


def _psi_from_counts(expected_counts, actual_counts):
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), EPS, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), EPS, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def calculate_psi(expected, actual, bins=10):
    """PSI of `actual` vs `expected`, using quantile bins of `expected`."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected_counts = np.histogram(expected, bins=edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(float)
    return _psi_from_counts(expected_counts, actual_counts)


def calculate_csi(train_features, test_features, feature_names, bins=10):
    """CSI per feature (same formula as PSI applied to each input feature)."""
    train_features = np.asarray(train_features, dtype=float)
    test_features = np.asarray(test_features, dtype=float)
    return {
        name: calculate_psi(train_features[:, i], test_features[:, i], bins)
        for i, name in enumerate(feature_names)
    }


def _status(value):
    if value < 0.1:
        return "Stable"
    if value < 0.25:
        return "Moderate shift"
    return "Significant shift"


def run_drift_analysis(
    train_proba, test_proba, train_features, test_features, feature_names, output_dir
):
    """Writes PSI (fraud_proba) and CSI (each feature) to PSI_CSI_analysis.csv."""
    rows = [
        {
            "metric": "PSI",
            "name": "fraud_proba",
            "value": calculate_psi(train_proba, test_proba),
        }
    ]
    for name, value in calculate_csi(train_features, test_features, feature_names).items():
        rows.append({"metric": "CSI", "name": name, "value": value})

    result = pd.DataFrame(rows)
    result["status"] = result["value"].apply(_status)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "PSI_CSI_analysis.csv"
    result.to_csv(path, index=False)
    print(f"Saved PSI/CSI analysis to {path}")
    return result
