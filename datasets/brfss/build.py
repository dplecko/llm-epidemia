
from pathlib import Path
out_path = Path(__file__).parent / "data" / "brfss.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import pandas as pd
import tempfile
import requests
import zipfile
import os
import numpy as np
import miceforest as mf

# Create temp folder
tmp_dir = tempfile.mkdtemp()

# Download SCF zip file
url = "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"
zip_path = os.path.join(tmp_dir, "scf2022s.zip")

with requests.get(url, stream=True) as r:
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Unzip contents
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

dta_path = os.path.join(tmp_dir, "LLCP2023.XPT ")
df = pd.read_sas(dta_path)

# Add state mapping
state_map = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California", 8: "Colorado",
    9: "Connecticut", 10: "Delaware", 11: "District of Columbia", 12: "Florida", 13: "Georgia",
    15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa", 20: "Kansas",
    22: "Louisiana", 23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana", 31: "Nebraska",
    32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico", 36: "New York",
    37: "North Carolina", 38: "North Dakota", 39: "Ohio", 40: "Oklahoma", 41: "Oregon",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming", 66: "Guam", 72: "Puerto Rico",
    78: "Virgin Islands"
}

df["state"] = df["_STATE"].map(state_map).astype("category")

df["sex"] = df["_SEX"].map({1.0: "Male", 2.0: "Female"}).astype("category")

df["race"] = df["_IMPRACE"].map({
    1.0: "White",
    2.0: "Black",
    3.0: "Asian",
    4.0: "AIAN",
    5.0: "Hispanic",
    6.0: "Other"
}).astype("category")

df["age"] = df["_AGE80"]

df["education"] = df["_EDUCAG"].map({
    1.0: "no high school",
    2.0: "high school",
    3.0: "some college",
    4.0: "college graduate",
    9.0: np.nan
}).astype("category")

df["income"] = df["_INCOMG1"].map({
    1.0: "<$15k",
    2.0: "$15–25k",
    3.0: "$25–35k",
    4.0: "$35–50k",
    5.0: "$50–100k",
    6.0: "$100k-200k",
    7.0: ">$200k",
    9.0: np.nan
}).astype("category")

# Additional health outcomes and variables
df["smoker"] = df["_SMOKER3"].map({
    1.0: "Yes", 2.0: "Yes", 3.0: "No", 4.0: "No", 9.0: np.nan
}).astype("category")

df["bmi"] = df["_BMI5"] / 100.0

df["exercise_monthly"] = df["EXERANY2"].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category")

df["poor_mental_health"] = df["_MENT14D"].map({1.0: "No", 2.0: "Yes", 3.0: "Yes", 9.0: np.nan}).astype("category")

df["diabetes"] = df["DIABETE4"].map({
    1.0: "Yes", 
    2.0: "Yes", # but during pregnancy 
    3.0: "No", 
    4.0: "No", 
    7.0: "No", 
    9.0: "No"
}).astype("category")

df["high_bp"] = df["BPHIGH6"].map({
    1.0: "Yes", 2.0: "Yes", # but pregnancy 
    3.0: "No",
    4.0: "No", 
    7.0: np.nan, 
    9.0: np.nan
}).astype("category")

df["asthma"] = df["ASTHMA3"].map({
    1.0: "Yes",
    2.0: "No", 
    4.0: "No", 
    7.0: "No", 
    9.0: "No"
}).astype("category")

df["cholesterol"] = df["TOLDHI3"].map({
    1.0: "Yes", 
    2.0: "No", 
    7.0: "No", 
    9.0: "No"
}).astype("category")

# Additional outcomes
df["heart_attack"] = df["CVDINFR4"].map({
    1.0: "Yes",
    2.0: "No",
    7.0: "No",
    9.0: "No"
}).astype("category")

df["stroke"] = df["CVDSTRK3"].map({
    1.0: "Yes",
    2.0: "No",
    7.0: "No",
    9.0: "No"
}).astype("category")

df["depression"] = df["ADDEPEV3"].map({
    1.0: "Yes",
    2.0: "No",
    7.0: "No",
    9.0: "No"
}).astype("category")

# deaf or serious difficulty
df["deaf"] = df["DEAF"].map({
    1.0: "Yes",
    2.0: "No",
    7.0: "No",
    9.0: "No"
}).astype("category")

# blind or serious difficulty
df["blind"] = df["BLIND"].map({
    1.0: "Yes",
    2.0: "No",
    7.0: "No",
    9.0: "No"
}).astype("category")

df["weight"] = df["_LLCPWT"]

df = df[[
    "state", "sex", "race", "age", "education", "income",
    "smoker", "bmi", "exercise_monthly", "poor_mental_health", "diabetes",
    "high_bp", "asthma", "cholesterol", "heart_attack", "stroke",
    "depression", "deaf", "blind", "weight"
]]

df = df.reset_index(drop=True)
kds = mf.ImputationKernel(df, save_all_iterations_data=True, random_state=0)
kds.mice(5)
df_impute = kds.complete_data()

df_impute.to_parquet(out_path, index=False)