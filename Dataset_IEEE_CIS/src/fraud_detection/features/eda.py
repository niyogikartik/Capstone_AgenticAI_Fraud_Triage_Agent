from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fraud_detection.settings import get_project_paths
from fraud_detection.utils.io import ensure_dir


@dataclass
class EDAArtifacts:
    summary_csv: Path
    missingness_csv: Path
    target_plot_png: Path



def run_basic_eda(df: pd.DataFrame, target_col: str = "isFraud") -> EDAArtifacts:
    paths = get_project_paths()
    out_dir = ensure_dir(paths.artifacts_root / "eda")

    summary_path = out_dir / "describe.csv"
    missing_path = out_dir / "missingness.csv"
    target_plot_path = out_dir / "target_distribution.png"

    df.describe(include="all").transpose().to_csv(summary_path)

    missing_df = (
        df.isna().mean().sort_values(ascending=False).rename("missing_ratio").to_frame().reset_index()
    )
    missing_df.columns = ["feature", "missing_ratio"]
    missing_df.to_csv(missing_path, index=False)

    if target_col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[target_col])
        plt.title("Fraud Label Distribution")
        plt.tight_layout()
        plt.savefig(target_plot_path)
        plt.close()

    return EDAArtifacts(
        summary_csv=summary_path,
        missingness_csv=missing_path,
        target_plot_png=target_plot_path,
    )
