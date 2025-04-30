

from pathlib import Path
out_path = Path(__file__).parent / "data" / "labor.parquet"
if out_path.exists():
    print("Parquet exists. Skipping."); exit()

import pandas as pd
import re
from textblob import Word
import sys, os
sys.path.append(os.path.abspath("datasets"))
from helpers import split_counts
import requests
from io import BytesIO

# direct download no longer works
url = "https://www.bls.gov/cps/cpsaat11.xlsx"
df = pd.read_excel(url, sheet_name="cpsaat11", skiprows=8)

# direct download no longer works
in_path = Path(__file__).parent / "data" / "cpsaat11.xlsx"
in_path = "datasets/labor/data/cpsaat11.xlsx"
df = pd.read_excel(in_path, skiprows=8, engine="openpyxl")

# Select relevant columns
df = df.iloc[:, 0:6]

df.columns = ["occupation", "total", "female", "white", "black", "asian"]

# Replace en-dash and encoding
df.replace({"\u2013": "-", "‚Äì": "-"}, inplace=True)
df = df.map(lambda x: x if x not in ["-", "–", "‚Äì"] else None)  # Convert to NaN

df["other"] = 100 - df["white"] - df["black"] - df["asian"]
df["female"] = df["female"] / 100
df["male"] = 1 - df["female"]

for col in ["white", "black", "asian", "other"]:
    df[col] = df[col] * df["total"] / 100

df = split_counts(df, "occupation", ["female", "male"], ["white", "black", "asian", "other"])

# Drop rows where occupation is NaN
df = df.dropna(subset=["occupation"])

rm_headings = [
    "Total, 16 years and over",
    "Management,  professional, and related occupations", 
    "Management, business, and financial operations occupations", 
    "Management occupations",
    "Business and financial operations occupations",
    "Professional and related occupations",
    "Computer and mathematical occupations",
    "Architecture and engineering occupations",
    "Life, physical, and social science occupations",
    "Community and social service occupations",
    "Legal occupations",
    "Education, training, and library occupations",
    "Arts, design, entertainment, sports, and media occupations",
    "Healthcare practitioners and technical occupations",
    "Health diagnosing and treating practitioners and other technical occupations",
    "Health technologists and technicians",
    "Service occupations",
    "Protective service occupations",
    "Food preparation and serving related occupations",
    "Building and grounds cleaning and maintenance occupations",
    "Personal care and service occupations",
    "Sales and related occupations",
    "Office and administrative support occupations",
    "Farming, fishing, and forestry occupations",
    "Construction and extraction occupations",
    "Installation, maintenance, and repair occupations",
    "Production occupations",
    "Transportation and material moving occupations",
    "Military specific occupations",
    "Unemployed, last worked 5 years ago or earlier",
    "Unemployed, never worked",
    "Not in labor force"
]
df = df[~df["occupation"].isin(rm_headings)]
df = df[~df["occupation"].str.contains(r"\ball other\b", case=False, na=False)]

# Remove numbers and unwanted characters from occupation names
df.loc[:, "occupation"] = df["occupation"].apply(lambda x: re.sub(r"[\d\-]+", "", str(x)).strip())

# Convert to lowercase
df["occupation"] = df["occupation"].str.lower()

manual_replacements = {
    "other drafters": None,
    "other designers": None,
    "other psychologists": None,
    "other healthcare practitioners and technical occupations": None,
    "other teachers and instructors": None,
    "other woodworkers": None,
    "other textile, apparel, and furnishings workers": None,
    "directors, religious activities and education": "religious program director",
    "agents and business managers of artists, performers, and athletes": "business manager for artists and athletes",
    "disc jockeys, except radio": "event disc jockey",
    "sailors and marine oilers": "maritime crew member",
    "credit examiners and collectors, and revenue agents": "credit examiner or collector",
    "speechlanguage pathologists": "speech-language pathologist",
    "helpersinstallation, maintenance, and repair workers": "installation, maintenance and repair helper",
    "doortodoor sales workers, news and street vendors, and related workers": "door-to-door sales worker or street vendor",
    "other community and social service specialists": None,
    "other educational instruction and library workers": None,
    "other protective service workers": None,
    "bus drivers, transit and intercity": "transit and intercity bus driver",
    "other entertainment attendants and related workers": None,
    "other production workers": None,
    "other material moving workers": None,
    "packer and packagers, hand": "hand packager",
}

df["occupation"] = df["occupation"].replace(manual_replacements)
df = df.dropna(subset=["occupation"])

# Remove plural forms from occupation names
df.loc[:, "occupation"] = df["occupation"].apply(lambda x: " ".join([Word(word).singularize() for word in x.split()]))

df = df.dropna()
df.to_parquet(out_path, index=False)
