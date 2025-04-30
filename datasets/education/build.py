
# raw dataset can be downloaded from:
# "https://nces.ed.gov/ipeds/SummaryTables/DownloadExcel/3601_2023_1_1_5ed94380-cc4e-494e-aae3-3293fcdde3c4"

from pathlib import Path
out_path = Path(__file__).parent / "data" / "education.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import pandas as pd

in_path = Path(__file__).parent / "data" / "education.xlsx"
df = pd.read_excel(in_path, skiprows=4)

df['Men'] = pd.to_numeric(df['Men'].str.replace(',', ''), errors='coerce')
df['Women'] = pd.to_numeric(df['Women'].str.replace(',', ''), errors='coerce')

# compute the percentage of male graduates in each degree
df['percent_male'] = 100 * df['Men'] / (df['Men'] + df['Women'])
df['percent_female'] = 100 - df['percent_male']

df['degree'] = df['CIP Title'].str.lower().str.strip()

df = df[['degree', 'percent_male', 'percent_female']]

df = df.melt(id_vars="degree", var_name="sex", value_name="weight")
df["sex"] = df["sex"].str.replace("percent_", "")
df.to_parquet(out_path, index=False)