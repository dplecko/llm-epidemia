
# source: https://www.bls.gov/cps/cpsaat11.xlsx
import pandas as pd
import re
from textblob import Word

# Load the XLSX file
file_path = "data/raw/labor/cpsaat11.xlsx"  # Adjust the path as needed
df = pd.read_excel(file_path, sheet_name="cpsaat11", skiprows=8)

# Select relevant columns
df = df.iloc[:, [0, 2, 3, 4, 5, 6]]
df.columns = ["occupation", "women", "white", "black_or_aa", "asian", "hispanic_or_latino"]

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
df["occupation"] = df["occupation"].apply(lambda x: re.sub(r"[\d\-]+", "", str(x)).strip())

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
df["occupation"] = df["occupation"].apply(lambda x: " ".join([Word(word).singularize() for word in x.split()]))

# Removes the last row which contains a note
df = df.iloc[:-1]  

# Replace en-dash and encoding
df.replace({"\u2013": "-", "‚Äì": "-"}, inplace=True)
df = df.applymap(lambda x: x if x not in ["-", "–", "‚Äì"] else None)  # Convert to NaN

# Fix the columns
df['percent_male'] = 100 - df['women']
df['percent_female'] = df['women']

# Subset the columns
df = df[['occupation', 'percent_male', 'percent_female']]

df = df.dropna()

df = df.melt(id_vars="occupation", var_name="sex", value_name="weight")
df["sex"] = df["sex"].str.replace("percent_", "")
df.to_parquet("data/clean/labor.parquet", index=False)
