from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized project paths used across modules."""

    project_root: Path
    data_root: Path
    artifacts_root: Path



DEFAULT_COLAB_DATA_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/IEEE_CIS_Dataset")



def get_project_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parents[2]

    env_data_root = os.getenv("FRAUD_DATA_ROOT")
    if env_data_root:
        data_root = Path(env_data_root)
    elif DEFAULT_COLAB_DATA_ROOT.exists():
        data_root = DEFAULT_COLAB_DATA_ROOT
    else:
        data_root = root / "ieee-fraud-detection"

    artifacts_root = root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(project_root=root, data_root=data_root, artifacts_root=artifacts_root)
