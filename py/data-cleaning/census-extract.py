
import pandas as pd
import requests
import numpy as np
from io import BytesIO
import pyarrow as pa
import pyarrow.parquet as pq

url = "https://api.census.gov/data/2023/acs/acs1/pums"
vars = [
    "PWGTP", "SEX", "AGEP", "RAC1P", "HISP", "NATIVITY", "CIT", "MAR",
    "NPF", "NOC", "ENG", "SCHL", "PERNP", "WAGP", "WKHP",
    "ESR", "COW", "OCCP", "INDP", "POWSP"
]
key = "1ddb0b9f7331dd07bcf3fc8b492d66c69d751cc4"

params = {
    "get": ",".join(vars),
    "for": "state:*",
    "key": key
}

resp = requests.get(url, params=params)
resp.raise_for_status()

data = resp.json()
df = pd.DataFrame(data[1:], columns=data[0])


# variable cleaning
cen = pd.DataFrame()
cen["weight"] = pd.to_numeric(df["PWGTP"])
cen["state"] = df["state"].astype("category")

cen["sex"] = pd.Series(np.where(df["SEX"] == "2", "female", "male")).astype("category")
cen["age"] = pd.to_numeric(df["AGEP"])

(all(cen["sex"] == pd.read_parquet("data/clean/census.parquet")["sex"]))

race_map = {
    "1": "white", "2": "black", "3": "AIAN", "4": "AIAN", "5": "AIAN",
    "6": "asian", "7": "NHOPI", "8": "other", "9": "mix"
}
cen["race"] = df["RAC1P"].map(race_map).astype("category")

cen["hispanic_origin"] = pd.Series(np.where(df["HISP"] != "01", "yes", "no")).astype("category")

cen["nativity"] = df["NATIVITY"].map({"1": "native", "2": "foreign-born"}).astype("category")

cit_map = {
    "1": "born in the US",
    "2": "born in Puerto Rico",
    "3": "born abroad of American parents",
    "4": "naturalized citizen",
    "5": "not a citizen"
}
cen["citizenship"] = df["CIT"].map(cit_map).astype("category")

mar_map = {
    "1": "married",
    "2": "widowed",
    "3": "divorced",
    "4": "separated",
    "5": "never married"
}
cen["marital"] = df["MAR"].map(mar_map).astype("category")

cen["family_size"] = pd.to_numeric(df["NPF"])
cen["children"] = pd.to_numeric(df["NOC"])
cen["english_level"] = pd.to_numeric(df["ENG"])

edu_level = pd.to_numeric(df["SCHL"])
cen["education_level"] = edu_level
cen["education_level"] = cen["education_level"].replace(0, np.nan)

edu_map = [
    'No schooling completed', 'Nursery school, preschool', 'Kindergarten', 'Grade 1',
    'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8',
    'Grade 9', 'Grade 10', 'Grade 11', '12th grade - no diploma',
    'Regular high school diploma', 'GED or alternative credential',
    'Some college, but less than 1 year',
    '1 or more years of college credit, no degree', "Associate's degree",
    "Bachelor's degree", "Master's degree",
    "Professional degree beyond a bachelor's degree", 'Doctorate degree'
]

cen["education"] = cen["education_level"].apply(lambda x: edu_map[int(x) - 1] if pd.notna(x) else np.nan)

cen["earnings"] = pd.to_numeric(df["PERNP"], errors="coerce")
cen["salary"] = pd.to_numeric(df["WAGP"], errors="coerce").replace(-1, np.nan)
cen["hours_worked"] = pd.to_numeric(df["WKHP"], errors="coerce")

esr_map = {
    "1": "employed",
    "2": "not at work",
    "3": "unemployed",
    "4": "employed",
    "5": "not at work",
    "6": "not in labor force"
}
cen["employment_status"] = df["ESR"].map(esr_map).astype("category")

cow_map = {
    "1": "for-profit company",
    "2": "non-profit company",
    "3": "government",
    "4": "government",
    "5": "government",
    "6": "self-employed",
    "7": "self-employed",
    "8": "self-employed"
}
cen["employer"] = df["COW"].map(cow_map).astype("category")

# occupation levels mapping
occ_url = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/code_lists/ACSPUMS2023CodeLists.xls"
occ_xls = BytesIO(requests.get(occ_url).content)
occ_tab = pd.read_excel(occ_xls, sheet_name="Occupation", header=None, skiprows=10)

cen["occupation"] = df["OCCP"].astype(str)
cen.loc[cen["occupation"] == "0009", "occupation"] = np.nan

cen["occp_name"] = ""
cen["occp_code"] = ""

for _, row in occ_tab.iterrows():
    key = str(row[0]).strip()
    if not key or key == "nan":
        continue
    match = cen["occupation"] == key
    cen.loc[match, "occp_name"] = row[2]
    cen.loc[match, "occp_code"] = str(row[1])
    
cen["occp_code"] = cen["occp_code"].astype("category")
cen = cen.drop(columns="occupation")

# industry levels mapping
ind_tab = pd.read_excel(occ_xls, sheet_name="Industry", header=None, skiprows=20)

cen["industry"] = df["INDP"].astype(str)
cen.loc[cen["industry"] == "0169", "industry"] = np.nan

cen["ind_name"] = ""
cen["ind_code"] = ""

for _, row in ind_tab.iterrows():
    key = str(row[2]).strip()
    if not key or key == "nan":
        continue
    match = cen["industry"] == key
    cen.loc[match, "ind_name"] = row[1]
    cen.loc[match, "ind_code"] = str(row[4])

cen["ind_code"] = cen["ind_code"].astype("category")
cen = cen.drop(columns="industry")

cen["place_of_work"] = df["POWSP"].replace("N", np.nan).astype("category")
cen["economic_region"] = np.nan

cen.loc[cen["place_of_work"].isin(['009', '023', '025', '033', '044', '050']), "economic_region"] = "New England"
cen.loc[cen["place_of_work"].isin(['010', '011', '024', '034', '036', '042']), "economic_region"] = "Mideast"
cen.loc[cen["place_of_work"].isin(['017', '018', '026', '039', '055']), "economic_region"] = "Great Lakes"
cen.loc[cen["place_of_work"].isin(['019', '020', '027', '029', '031', '038', '046']), "economic_region"] = "Plains"
cen.loc[cen["place_of_work"].isin(['001', '005', '012', '013', '021', '022', '028', '037', '045', '047', '051', '054']), "economic_region"] = "Southeast"
cen.loc[cen["place_of_work"].isin(['004', '035', '040', '048']), "economic_region"] = "Southwest"
cen.loc[cen["place_of_work"].isin(['008', '016', '040', '048', '030', '049', '056']), "economic_region"] = "Rocky Mountain"
cen.loc[cen["place_of_work"].isin(['002', '006', '015', '032', '041', '053']), "economic_region"] = "Far West"

# Handle numeric codes > 56
cen.loc[pd.to_numeric(cen["place_of_work"], errors="coerce") > 56, "economic_region"] = "Abroad"

cen["economic_region"] = cen["economic_region"].astype("category")

ordering = [
    'sex', 'age',
    'race', 'hispanic_origin', 'citizenship', 'nativity',
    'marital', 'family_size', 'children', 'state',
    'education', 'education_level', 'english_level',
    'salary', 'hours_worked', 'employment_status',
    'occp_name', 'occp_code', 'ind_name', 'ind_code',
    'employer', 'place_of_work', 'economic_region',
    'weight'
]

cen = cen[ordering]

# save to parquet
cen.to_parquet("data/clean/census_py.parquet", index=False)
