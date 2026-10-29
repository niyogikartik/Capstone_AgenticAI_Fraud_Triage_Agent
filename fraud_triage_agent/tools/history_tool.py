"""SHAP-space nearest-neighbour search over validation transactions."""
from functools import lru_cache

import faiss
import numpy as np

from .. import config
from ..schemas.outputs import HistoricalReport, SimilarCase
from . import data_loader


def _normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows (float32) so inner product equals cosine similarity; zero rows stay zero."""
    x = np.ascontiguousarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


@lru_cache(maxsize=1)
def _index(cols: tuple[str, ...]) -> faiss.IndexFlatIP:
    """Build (once) an exact FAISS inner-product index over the normalised validation SHAP vectors."""
    vectors = _normalize(data_loader.validation()[list(cols)].values)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion (k successes out of n).

    Gives an honest range for "8 of 10 were fraud" when the sample is small.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def find_similar_history(transaction_id: int, k: int = config.NEIGHBORS_K) -> HistoricalReport:
    """Find the k validation transactions whose SHAP pattern is closest (cosine distance) to this
    transaction, drop matches beyond MAX_NEIGHBOR_DISTANCE, and count how many were TP vs FP."""
    hr, val = data_loader.high_risk(), data_loader.validation()
    cols = [c for c in data_loader.shap_columns(hr) if c in val.columns]
    q = hr.loc[hr[config.ID_COLUMN] == transaction_id, cols]
    if q.empty:
        raise KeyError(f"Transaction {transaction_id} not in high-risk set")
    index = _index(tuple(cols))
    sim, idx = index.search(_normalize(q.values), min(k, index.ntotal))
    dist = 1.0 - sim
    mask = dist[0] <= config.MAX_NEIGHBOR_DISTANCE
    kept = val.iloc[idx[0][mask]]
    labels = kept[config.LABEL_COLUMN]
    ids = kept["Val_Index"] if "Val_Index" in kept.columns else kept.index.to_series()
    matches = [
        SimilarCase(val_index=int(i), category=str(c), distance=float(d))
        for i, c, d in zip(ids, labels, dist[0][mask])
    ]
    n_tp, n_fp = int((labels == "TP").sum()), int((labels == "FP").sum())
    n = n_tp + n_fp
    lo, hi = _wilson(n_tp, n)
    return HistoricalReport(transaction_id=transaction_id, k=int(mask.sum()), n_tp=n_tp,
                            n_fp=n_fp, tp_rate=n_tp / n if n else 0.0,
                            tp_rate_ci_low=lo, tp_rate_ci_high=hi,
                            mean_distance=float(dist[0][mask].mean()) if mask.any() else 0.0,
                            matches=matches)
