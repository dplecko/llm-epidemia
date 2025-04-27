
import pandas as pd
from pathlib import Path

def load_meps():
    path = Path(__file__).parent / "data" / "meps.parquet"
    return pd.read_parquet(path)
