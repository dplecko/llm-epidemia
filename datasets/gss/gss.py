
import pandas as pd
from pathlib import Path

def load_gss():
    path = Path(__file__).parent / "data" / "gss.parquet"
    return pd.read_parquet(path)
