
import pandas as pd
import numpy as np
import zipfile
import tempfile
import requests
import pyreadstat
from sklearn.impute import SimpleImputer
import pyarrow.parquet as pq
import pyarrow as pa
import os
import miceforest as mf

# Download and unzip
url = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/2022_stata.zip"
tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
tmp_dir = tempfile.mkdtemp()

r = requests.get(url)
with open(tmp_zip.name, "wb") as f:
    f.write(r.content)

with zipfile.ZipFile(tmp_zip.name, "r") as zip_ref:
    zip_ref.extractall(tmp_dir)

df, meta = pyreadstat.read_dta(os.path.join(tmp_dir, "2022", "GSS2022.dta"))

# Select columns
keep = ["sex", "age", "race", "educ", "degree", "income", 
        "polviews", "partyid", "wtssps", "wtssnrps"]

df = df[keep]
df = df[df["wtssnrps"].notna()]
print(df.isna().mean().sort_values() * 100)

gss = pd.DataFrame()
gss["wgh"] = df["wtssnrps"].astype(float)
gss["sex"] = df["sex"].map({1: "male", 2: "female"}).astype("category")
gss["age"] = pd.to_numeric(df["age"], errors="coerce")
gss["race"] = df["race"].map({1: "white", 2: "black", 3: "other"}).astype("category")
gss["degree"] = df["degree"].map({
    0: "no high school", 1: "high school", 2: "junior college",
    3: "bachelor", 4: "graduate"
}).astype("category")
gss["edu"] = pd.to_numeric(df["educ"], errors="coerce")

gss["view"] = df["polviews"].map({
    1: "liberal", 2: "liberal", 3: "liberal", 4: "moderate",
    5: "conservative", 6: "conservative", 7: "conservative"
}).astype("category")

gss["party"] = df["partyid"].map({
    0: "democrat", 1: "democrat", 2: "democrat", 3: "independent",
    4: "republican", 5: "republican", 6: "republican", 7: np.nan
}).astype("category")

gss = gss.reset_index(drop=True)
kds = mf.ImputationKernel(gss, save_all_iterations_data=True, random_state=0)
kds.mice(5)
gss = kds.complete_data()

# Save to Parquet
gss.to_parquet("data/clean/gss_py.parquet", index=False)
