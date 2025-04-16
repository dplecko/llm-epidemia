
import pandas as pd

url = "https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/tables/table-42/table-42.xls"
df = pd.read_excel(url, sheet_name="19tbl42", skiprows=4, engine="xlrd")

# Select relevant columns
df = df.iloc[:, [0, 4, 5]]

# Rename columns
df.columns = ["crime_type", "percent_male", "percent_female"]

# Drop rows with missing crime types
df = df.dropna(subset=["crime_type"])

# Convert crime types to lowercase
df["crime_type"] = df["crime_type"].str.lower().str.strip()
df["crime_type"] = df["crime_type"].str.replace(r'\d+', '', regex=True).str.strip()

# Remove row for other offenses; Remove rows with notes
df = df[df["crime_type"] != "all other offenses (except traffic)"]
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


df = df.melt(id_vars="crime_type", var_name="sex", value_name="weight")
df["sex"] = df["sex"].str.replace("percent_", "")

df.to_parquet("data/clean/crime.parquet", index=False)
