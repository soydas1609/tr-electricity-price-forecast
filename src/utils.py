"""
utils.py
========
Shared utility functions.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).parent.parent


def load_processed() -> pd.DataFrame:
    """Load the processed hourly market dataset from parquet."""
    path = ROOT / "data" / "processed" / "hourly_market.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run `python src/data_loader.py` first."
        )
    return pd.read_parquet(path)
