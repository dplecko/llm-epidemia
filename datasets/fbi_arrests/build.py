
from pathlib import Path
out_path = Path(__file__).parent / "data" / "fbi_arrests.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import pandas as pd
import requests
from io import BytesIO
from helpers import split_counts

url = "https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/tables/table-42/table-42.xls/output.xls"
headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get(url, headers=headers)
df = pd.read_excel(BytesIO(r.content), sheet_name="19tbl42", skiprows=4)

url2 = "https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/tables/table-43/table-43a.xls/output.xls"
r2 = requests.get(url2, headers=headers)
df2 = pd.read_excel(BytesIO(r2.content), sheet_name=0, skiprows=6)

df = pd.merge(df, df2, on="Offense charged")

df = df.rename(columns={"Offense charged": "crime_type", "Percent\nmale": "male",
                        "Percent\nfemale": "female", 
                        "Black or\nAfrican\nAmerican": "Black",
                        "American\nIndian or\nAlaska\nNative": "AIAN",
                        "Native\nHawaiian\nor Other\nPacific\nIslander": "NHOPI"})

df["female"] = pd.to_numeric(df["female"], errors="coerce") / 100
df["male"] = pd.to_numeric(df["male"], errors="coerce") / 100
df = split_counts(df, "crime_type", ["male", "female"], ["White", "Black", "AIAN", "NHOPI", "Asian"])

# Drop rows with missing crime types
df = df.dropna(subset=["crime_type"])

# Convert crime types to lowercase
df["crime_type"] = df["crime_type"].str.lower().str.strip()
df["crime_type"] = df["crime_type"].str.replace(r'\d+', '', regex=True).str.strip()

# Remove row for other offenses; Remove rows with notes
df = df[~df["crime_type"].isin(["all other offenses (except traffic)", "because of rounding, the percentages may not add to .."])]
df = df.loc[:df[df["crime_type"] == "curfew and loitering law violations"].index[0]]

# Define manual replacements
manual_replacements = {
    "weapons; carrying, possessing, etc.": "carrying or possessing weapons",
    "other assaults": "assault",
    "larceny-theft": "theft",
    "stolen property; buying, receiving, possessing": "buying or receiving stolen property",
    "total": "a crime"
}

# Apply replacements
df["crime_type"] = df["crime_type"].replace(manual_replacements)

df.to_parquet(out_path, index=False)