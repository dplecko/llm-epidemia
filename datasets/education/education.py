
import pandas as pd
from pathlib import Path

def load_education():
    path = Path(__file__).parent / "data" / "education.parquet"
    return pd.read_parquet(path)
