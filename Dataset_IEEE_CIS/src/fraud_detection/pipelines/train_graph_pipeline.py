from __future__ import annotations

from fraud_detection.data.loader import build_training_frame, load_raw_data
from fraud_detection.models.graph_trainer import train_graph_features



def run(sample: bool = False, nrows: int | None = None) -> None:
    bundle = load_raw_data(sample=sample, nrows=nrows)
    train_df = build_training_frame(bundle)
    result = train_graph_features(train_df)

    print(f"Saved graph object: {result.graph_path}")
    print(f"Nodes: {result.node_count} | Edges: {result.edge_count}")
