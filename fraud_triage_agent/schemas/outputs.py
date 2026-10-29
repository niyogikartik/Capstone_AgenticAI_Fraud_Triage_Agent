from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """One feature's contribution to a transaction's fraud score."""

    feature: str
    shap_value: float
    feature_value: float | str | None = None
    direction: str  # "increases_risk" | "decreases_risk"


class ShapReport(BaseModel):
    """Output of the SHAP interpreter agent: probability plus the top (max 5) drivers."""

    transaction_id: int
    fraud_probability: float
    top_features: list[FeatureContribution] = Field(max_length=5)


class SimilarCase(BaseModel):
    """One retrieved historical validation case, so the analyst can look it up and verify it."""

    val_index: int  # row identifier in Validation_TP_FP_FN.csv (the file has no TransactionID)
    category: str  # TP | FP | FN
    distance: float  # cosine distance in SHAP space (0 = identical pattern)


class HistoricalReport(BaseModel):
    """Output of the historical look-up agent: how similar past cases turned out.

    k = matches used; tp_rate = n_tp / (n_tp + n_fp), with a 95% confidence interval.
    matches lists every retrieved case (closest first) for verification.
    """

    transaction_id: int
    k: int
    n_tp: int
    n_fp: int
    tp_rate: float
    tp_rate_ci_low: float
    tp_rate_ci_high: float
    mean_distance: float
    matches: list[SimilarCase] = Field(default_factory=list)


class DriftFeature(BaseModel):
    """Drift score (PSI/CSI) and status for one feature."""

    feature: str
    psi: float
    status: str  # stable | moderate | significant
    in_transaction_top_features: bool = False


class DriftReport(BaseModel):
    """Output of the model drift agent: worst overall status and the features worth showing."""

    overall_status: str
    drifting_features: list[DriftFeature]


class TriageSummary(BaseModel):
    """Final report returned to the analyst: summary text, optional LLM interpretation,
    each sub-agent's raw evidence, any caveats (failed or skipped sections) and a run_id."""

    transaction_id: int
    recommendation: str | None = None
    summary: str
    interpretation: str | None = None
    shap: ShapReport | None = None
    history: HistoricalReport | None = None
    drift: DriftReport | None = None
    caveats: list[str] = Field(default_factory=list)
    run_id: str | None = None
