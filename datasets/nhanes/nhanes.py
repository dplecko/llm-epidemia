
import pandas as pd
from pathlib import Path

def load_nhanes():
    path = Path(__file__).parent / "data" / "nhanes.parquet"
    return pd.read_parquet(path)
