
from pathlib import Path
out_path = Path(__file__).parent / "data" / "scf.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import tempfile
import requests
import zipfile
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
    1: "no high school", 2: "high school", 3: "some college", 4: "college degree"
})
df["married"] = df["married"].map({1: "yes", 2: "no"})

# Keep relevant columns
dem_cols = ["age", "sex", "race", "educ", "married", "kids"]

# outcomes
df["food"] = df["foodhome"] + df["foodaway"] + df["fooddelv"]

df["house_own"] = df["hhouses"].map({
    1: "yes", 0: "no"
})

df["weight"] = df["wgt"]

out_cols = ["food", "house_own", "income", "rent", "asset", "debt", "networth", "weight"]

df = df[dem_cols + out_cols]

df.to_parquet(out_path, index=False)