
import pandas as pd
from pathlib import Path

def load_fbi_arrests():
    path = Path(__file__).parent / "data" / "acs.parquet"
    return pd.read_parquet(path)
