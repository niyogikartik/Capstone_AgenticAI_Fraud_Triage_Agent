from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import pandas as pd

from fraud_detection.settings import get_project_paths
from fraud_detection.utils.io import ensure_dir


@dataclass
class GraphTrainingResult:
    graph_path: Path
    node_count: int
    edge_count: int



def build_entity_graph(
    df: pd.DataFrame,
    id_cols: Sequence[str] = ("card1", "card2", "addr1", "P_emaildomain"),
) -> nx.Graph:
    """Construct a heterogeneous relation graph from transaction entities."""
    graph = nx.Graph()
    valid_cols = [c for c in id_cols if c in df.columns]

    for _, row in df[valid_cols + ["TransactionID"]].dropna(subset=["TransactionID"]).iterrows():
        tx_node = f"tx:{int(row['TransactionID'])}"
        graph.add_node(tx_node, node_type="transaction")

        for col in valid_cols:
            val = row.get(col)
            if pd.notna(val):
                ent_node = f"{col}:{val}"
                graph.add_node(ent_node, node_type=col)
                graph.add_edge(tx_node, ent_node, relation=col)

    return graph



def train_graph_features(df: pd.DataFrame) -> GraphTrainingResult:
    paths = get_project_paths()
    out_dir = ensure_dir(paths.artifacts_root / "graphs")
    graph_path = out_dir / "entity_graph.gpickle"

    graph = build_entity_graph(df)
    nx.write_gpickle(graph, graph_path)

    return GraphTrainingResult(
        graph_path=graph_path,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
    )
