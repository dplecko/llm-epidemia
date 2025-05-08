
from pathlib import Path
out_path = Path(__file__).parent / "data" / "meps.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import pandas as pd
import requests
import sys, os
sys.path.append(os.path.join(os.getcwd(), "datasets"))
from helpers import discrete_col
from io import BytesIO
from zipfile import ZipFile
import miceforest as mf

# Download the zip file from the URL
url = "https://meps.ahrq.gov/mepsweb/data_files/pufs/h243/h243xlsx.zip"
response = requests.get(url)
response.raise_for_status()

# Extract the xlsx file from the zip
with ZipFile(BytesIO(response.content)) as z:
    xlsx_filename = [name for name in z.namelist() if name.endswith(".xlsx")][0]
    with z.open(xlsx_filename) as xlsx_file:
        df_full = pd.read_excel(xlsx_file)

#, sheet_name="19tbl42", skiprows=4)

# get a subset of columns
vars_meps = [
    # Demographics
    #"AGE31X",  # Age at round 1
    #"AGE42X",  # Age at round 2
    "AGE53X",  # Age at round 3
    "SEX",     # Gender
    "RACEV1X", # Race (Version 1)
    "RACETHX", # Race/ethnicity combined
    "EDUCYR",# Education level
    "REGION53",  # Census region
    "MARRY53X",# Marital status

    # Insurance
    "INSURC22",    # Any insurance coverage
    "PUB53X", # Public insurance
    "PRIV53",   # Private insurance
    "MCREV22",     # Medicare ever
    "MCDEV22",     # Medicaid ever

    # Expenditures
    "TOTEXP22",   # Total expenditures
    "OBVEXP22",   # Office-based expenditures
    "RXEXP22",    # Prescription expenditures
    "IPTEXP22", # Inpatient expenditures
    "DVTEXP22",    # Dental expenditures

    # Utilization
    "OBTOTV22",   # Office-based visits
    "OPTOTV22",    # Outpatient visits
    "IPDIS22",    # HOSPITAL DISCHARGES 2022
    "RXTOT22",    # PRESC MEDS INCL REFILLS 22
    "DVTOT22",  # Dental visits

    # Survey weights
    "PERWT22F", # Final person-level weight for 2022
    "VARSTR",   # Variance stratum
    "VARPSU"    # Primary sampling unit
]

df = df_full[vars_meps]

# Decode categorical variables
df["sex"] = df["SEX"].map({1: "male", 2: "female"}).astype("category")

race_map = {
    1: "white",
    2: "black",
    3: "AIAN",
    4: "asian",
    6: "multiple"
}
df["race"] = df["RACEV1X"].map(race_map).astype("category")

marry_map = {
    1: "married",
    2: "widowed",
    3: "divorced",
    4: "separated",
    5: "never married",
    6: "inapplicable",
    7: "married",
    8: "widowed",
    9: "divorced",
    10: "separated",
    -7: "refused",
    -1: "inapplicable",
}
df["marital"] = df["MARRY53X"].map(marry_map).astype("category")

region_map = {
    -1: "inapplicable",
    1: "northeast",
    2: "midwest",
    3: "south",
    4: "west"
}
df["region"] = df["REGION53"].map(region_map).astype("category")

insure_map = {
    **dict.fromkeys([1, 2, 4, 5, 6, 8], "yes"),
    **dict.fromkeys([3, 7], "no")
}
df["insured"] = df["INSURC22"].map(insure_map).astype("category")

medicare_map = {
    1: "yes",
    2: "no"
}
df

insure_type_map = {
    1: "private",
    2: "public",
    3: "none",
    4: "medicare",
    5: "private",
    6: "public",
    7: "none",
    8: "other"
}
df["insure_type"] = df["INSURC22"].map(insure_type_map).astype("category")

df["age"] = df["AGE53X"]
df["education_years"] = df["EDUCYR"]
df["expenditure"] = df["TOTEXP22"]
df["outpatient_visits"] = df["OPTOTV22"]
df["office_visits"] = df["OBTOTV22"]
df["inpatient_visits"] = df["IPDIS22"]
df["dental_visits"] = df["DVTOT22"]
df["weight"] = df["PERWT22F"]

df["age_group"] = pd.cut(
    df["age"],
    bins=[-1, 17, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, float("inf")],
    labels=[
        "<18", "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"
    ],
    right=True
)

df = discrete_col(
    df, col="expenditure", breaks=[200, 1000, 1500, 5000, 10000, 30000], 
    unit="US dollars",
    last_plus=True
)

def visits_discrete(df, col="office_visits"):
    new_col=col+"_group"
    # Define the bins and labels directly
    bins = list(range(0, 11)) + [float("inf")]
    labels = [str(i) for i in range(0, 10)] + ["10+"]

    # Add the grouped column
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels, right=True, include_lowest=True)
    return df

for col in ["office_visits", "inpatient_visits", "dental_visits"]:
    df = visits_discrete(df, col=col)

print(df.drop(columns=vars_meps).isna().mean().sort_values() * 100)

df = df.drop(columns=vars_meps)

# imputation for 195 missing values in the age_group column
df = df.reset_index(drop=True)
exclude_cols = ["age"]
cols = df.columns.tolist()
variable_schema = {
    col: [c for c in cols if c != col and c not in exclude_cols]
    for col in cols
    if col not in exclude_cols
}
kds = mf.ImputationKernel(df, save_all_iterations_data=True, random_state=0, variable_schema=variable_schema)
kds.mice(5)
df = kds.complete_data()

df.to_parquet(out_path, index=False)