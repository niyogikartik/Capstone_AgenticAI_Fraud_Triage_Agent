from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV with consistent defaults."""
    return pd.read_csv(path, nrows=nrows)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
