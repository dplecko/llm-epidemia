
import pandas as pd
from pathlib import Path

def load_labor():
    path = Path(__file__).parent / "data" / "labor.parquet"
    return pd.read_parquet(path)
