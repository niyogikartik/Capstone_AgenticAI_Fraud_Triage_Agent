from __future__ import annotations

from fraud_detection.data.loader import build_training_frame, load_raw_data
from fraud_detection.features.eda import run_basic_eda



def run(sample: bool = True, nrows: int | None = 100000) -> None:
    bundle = load_raw_data(sample=sample, nrows=nrows)
    train_df = build_training_frame(bundle)

    artifacts = run_basic_eda(train_df)
    print(f"Summary saved to: {artifacts.summary_csv}")
    print(f"Missingness saved to: {artifacts.missingness_csv}")
    print(f"Target plot saved to: {artifacts.target_plot_png}")
