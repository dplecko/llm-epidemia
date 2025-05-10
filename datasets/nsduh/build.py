
from pathlib import Path
out_path = Path(__file__).parent / "data" / "nsduh.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import pandas as pd
import tempfile
import requests
import zipfile
import os

# Create temp folder
tmp_dir = tempfile.mkdtemp()

# Download SCF zip file
url = "https://www.samhsa.gov/data/system/files/media-puf-file/NSDUH-2023-DS0001-bndl-data-stata_v1.zip"
zip_path = os.path.join(tmp_dir, "scf2022s.zip")

with requests.get(url, stream=True) as r:
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Unzip contents
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

dta_path = os.path.join(tmp_dir, "NSDUH_2023.dta")
df = pd.read_stata(dta_path)

age_map = {
    '1 - Respondent is 12 or 13 years old': "12–13 years",
    '2 - Respondent is 14 or 15 years old': "14–15 years",
    '3 - Respondent is 16 or 17 years old': "16–17 years",
    '4 - Respondent is between 18 and 20 years old': "18–20 years",
    '5 - Respondent is between 21 and 23 years old': "21–23 years",
    '6 - Respondent is 24 or 25 years old': "24–25 years",
    '7 - Respondent is between 26 and 29 years old': "26–29 years",
    '8 - Respondent is between 30 and 34 years old': "30–34 years",
    '9 - Respondent is between 35 and 49 years old': "35–49 years",
    '10 - Respondent is between 50 and 64 years old': "50-64 years",
    '11 - Respondent is 65 years old or older': "65+ years"
}
df["age"] = df["AGE3"].map(age_map).astype("category")
df["sex"] = df["irsex"].map(lambda x: "male" if x == "1 - Male" else "female").astype("category")

race_map = {
    '1 - NonHisp White': "White",
    '2 - NonHisp Black/Afr Am': "Black",
    '3 - NonHisp Native Am/AK Native': "Native American",
    '4 - NonHisp Native HI/Other Pac Isl': "Pacific Islander",
    '5 - NonHisp Asian': "Asian",
    '6 - NonHisp more than one race': "Multiple",
    '7 - Hispanic': "Hispanic"
}
df["race"] = df["NEWRACE2"].map(race_map).astype("category")

edu_map = {
    '1 - Fifth grade or less grade completed': "≤ 8th grade",
    '2 - Sixth grade completed': "≤ 8th grade",
    '3 - Seventh grade completed': "≤ 8th grade",
    '4 - Eighth grade completed': "≤ 8th grade",
    '5 - Ninth grade completed': "Some high school",
    '6 - Tenth grade completed': "Some high school",
    '7 - Eleventh or Twelfth grade completed, no diploma': "Some high school",
    '8 - High school diploma/GED': "High school graduate",
    '9 - Some college credit, but no degree': "Some college, no degree",
    '10 - Associate s degree': "Associate degree",
    '11 - College graduate or higher': "Bachelor’s or higher"
}
df["edu"] = df["IREDUHIGHST2"].map(edu_map).astype("category")

# ever variables
df["alc_ever"] = df["alcever"].map(lambda x: "yes" if x == "1 - Yes" else "no").astype("category")
df["cig_ever"] = df["cigever"].map(lambda x: "yes" if x == "1 - Yes" else "no").astype("category")
df["mj_ever"] = df["mjever"].map(lambda x: "yes" if x == "1 - Yes" else "no").astype("category")
df["coc_ever"] = df["cocever"].map(lambda x: "yes" if x == "1 - Yes" else "no").astype("category")
df["her_ever"] = df["herever"].map(lambda x: "yes" if x == "1 - Yes" else "no").astype("category")

# monthly variables
df["alc_monthly"] = df["alcrec"].map(lambda x: "yes" if x == "1 - Within the past 30 days" else "no").astype("category")
df["cig_monthly"] = df["cigrec"].map(lambda x: "yes" if x == "1 - Within the past 30 days" else "no").astype("category")
df["mj_monthly"] = df["mjrec"].map(lambda x: "yes" if x == "1 - Within the past 30 days" else "no").astype("category")
df["coc_monthly"] = df["cocrec"].map(lambda x: "yes" if x == "1 - Within the past 30 days" else "no").astype("category")
df["her_monthly"] = df["herrec"].map(lambda x: "yes" if x == "1 - Within the past 30 days" else "no").astype("category")

# weight column
df["weight"] = df["ANALWT2_C"]

vars_created = [
    "age", "sex", "race", "edu",
    "alc_ever", "cig_ever", "mj_ever", "coc_ever", "her_ever",
    "alc_monthly", "cig_monthly", "mj_monthly", "coc_monthly", "her_monthly",
    "weight"
]

df = df[vars_created]
df.to_parquet(out_path, index=False)