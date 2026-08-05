from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from fraud_detection.settings import get_project_paths
from fraud_detection.utils.io import read_csv


@dataclass
class DataBundle:
    train_transaction: pd.DataFrame
    train_identity: pd.DataFrame
    test_transaction: pd.DataFrame
    test_identity: pd.DataFrame



def load_raw_data(sample: bool = False, nrows: Optional[int] = None) -> DataBundle:
    paths = get_project_paths()
    if sample:
        train_tx_path = paths.project_root / "train_transaction_sample.csv"
        train_id_path = paths.project_root / "train_identity_sample.csv"
    else:
        train_tx_path = paths.data_root / "train_transaction.csv"
        train_id_path = paths.data_root / "train_identity.csv"

    test_tx_path = paths.data_root / "test_transaction.csv"
    test_id_path = paths.data_root / "test_identity.csv"

    return DataBundle(
        train_transaction=read_csv(train_tx_path, nrows=nrows),
        train_identity=read_csv(train_id_path, nrows=nrows),
        test_transaction=read_csv(test_tx_path, nrows=nrows),
        test_identity=read_csv(test_id_path, nrows=nrows),
    )


def build_training_frame(bundle: DataBundle) -> pd.DataFrame:
    return bundle.train_transaction.merge(bundle.train_identity, on="TransactionID", how="left")
