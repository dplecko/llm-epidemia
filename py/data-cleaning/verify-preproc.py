
import pandas as pd
import numpy as np

df_old = pd.read_parquet("data/clean/census.parquet")
df_new = pd.read_parquet("data/clean/census_py.parquet")

assert set(df_old.columns) == set(df_new.columns), "Column sets differ"

for col in df_old.columns:
    a = df_old[col]
    b = df_new[col]

    # Align types for comparison
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        # Convert both to float64 and compare with NaNs aligned
        a = pd.to_numeric(a, errors="coerce").astype(float)
        b = pd.to_numeric(b, errors="coerce").astype(float)
        equal = np.allclose(a.fillna(-99999), b.fillna(-99999))
    else:
        # Compare as strings with NA alignment
        a_str = a.astype(str).str.strip().fillna("NA")
        b_str = b.astype(str).str.strip().fillna("NA")
        equal = a_str.equals(b_str)

    if not equal:
        print(f"❌ Mismatch in column: {col}")
        mask = a_str != b_str
        diff = pd.DataFrame({'old': a_str[mask], 'new': b_str[mask]})
        print(diff.head(10))
    else:
        print(f"✅ {col} matches")