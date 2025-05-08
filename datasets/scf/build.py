
from pathlib import Path
out_path = Path(__file__).parent / "data" / "scf.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import tempfile
import requests
import zipfile
import sys, os
sys.path.append(os.path.join(os.getcwd(), "datasets"))
from helpers import discrete_col
import os
import pandas as pd

# Create temp folder
tmp_dir = tempfile.mkdtemp()

# Download SCF zip file
url = "https://www.federalreserve.gov/econres/files/scfp2022s.zip"
zip_path = os.path.join(tmp_dir, "scfp2022s.zip")

with requests.get(url, stream=True) as r:
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Unzip contents
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

# List contents
print("Extracted to:", tmp_dir)
print(os.listdir(tmp_dir))

dta_path = os.path.join(tmp_dir, "rscfp2022.dta")
df = pd.read_stata(dta_path)

# demographic variables
df["sex"] = df["hhsex"].map({1: "Male", 2: "Female"})
df["race"] = df["racecl5"].map({
    1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"
})
df["education"] = df["edcl"].map({
    1: "no high school", 2: "high school", 3: "some college", 4: "a college degree"
})
df["married"] = df["married"].map({1: "yes", 2: "no"})

df["age_group"] = pd.cut(
    df["age"],
    bins=[-1, 17, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, float("inf")],
    labels=[
        "<18", "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"
    ],
    right=True
)

# Keep relevant columns
dem_cols = ["age", "age_group", "sex", "race", "education", "married", "kids"]

# outcomes
df["food"] = df["foodhome"] + df["foodaway"] + df["fooddelv"]

# discretize columns food, asset, debt, networth
df = discrete_col(
    df, col="food", breaks=[3500, 5500, 7000, 8500, 10000, 15000, 30000], 
    unit="US dollars",
    last_plus=True
)

df = discrete_col(
    df, col="asset", breaks=[10e4, 3 * 10e4, 10e5, 3 * 10e5, 10e6, 10e7], 
    unit="US dollars",
    last_plus=True
)

for col in ["debt", "networth"]:
    df = discrete_col(
        df, col=col, breaks=[1000, 10000, 30000, 10e5, 3 * 10e5, 10e6], 
        unit="US dollars",
        last_plus=True
    )

df["house_own"] = df["hhouses"].map({
    1: "yes", 0: "no"
})

df["weight"] = df["wgt"]

out_cols = ["food", "food_group", "house_own", "income", "rent", 
            "asset", "asset_group", "debt", "debt_group", "networth", "networth_group", "weight"]

df = df[dem_cols + out_cols]

df.to_parquet(out_path, index=False)