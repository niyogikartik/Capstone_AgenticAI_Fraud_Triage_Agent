import pandas as pd

from fraud_detection.features.preprocessing import basic_feature_cleanup



def test_basic_feature_cleanup_adds_amount_log() -> None:
    df = pd.DataFrame({"TransactionAmt": [100.0, 0.0], "TransactionDT": [86400, 172800]})
    out = basic_feature_cleanup(df)

    assert "TransactionAmt_log1p" in out.columns
    assert "TransactionDay" in out.columns
