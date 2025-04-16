
# source: https://wwwn.cdc.gov/NHISDataQueryTool/SHS_adult/index.html
import pandas as pd
import os

# Define paths
input_folder = "data/raw/health"
output_file = "data/clean/health.csv"

# US Adult Population Estimates (2023)
us_population_total = 166_000_000
us_population_male = us_population_total * 0.49  # 49% male
us_population_female = us_population_total * 0.51  # 51% female

# List of disease CSVs
disease_files = [
    "angina.csv", "arthritis.csv", "asthma.csv", "breast-cancer.csv",
    "chd.csv", "cholesterol.csv", "chronic-pain.csv", "copd.csv",
    "current-asthma.csv", "diabetes.csv"
]

# Function to process each disease file correctly
def process_disease(file_path, disease_name):
    try:
        # Read the entire file as raw lines
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Extract relevant rows (percentages and confidence intervals)
        percentage_row = lines[2].strip().split(",")  # Third line (index 2) for percentages
        ci_row = lines[6].strip().split(",")  # Seventh line (index 6) for confidence intervals

        # Convert extracted values to numeric, keeping only the first number in a range
        year = int(percentage_row[0].strip('"'))  # Extract year

        female_incid = float(percentage_row[1].strip('"').split(",")[0])
        male_incid = float(percentage_row[2].strip('"').split(",")[0])

        female_ci = float(ci_row[1].strip('"').split(",")[0])
        male_ci = float(ci_row[2].strip('"').split(",")[0])

        # Compute total number of cases
        female_cases = (female_incid / 100) * us_population_female
        male_cases = (male_incid / 100) * us_population_male

        # Compute male/female proportions
        percent_male = 100 * male_cases / (male_cases + female_cases)
        percent_female = 100 * female_cases / (male_cases + female_cases)

        # Create dataframe
        df_combined = pd.DataFrame([{
            "disease": disease_name,
            "female_incid": female_incid,
            "male_incid": male_incid,
            "female_ci": female_ci,
            "male_ci": male_ci,
            "percent_male": percent_male,
            "percent_female": percent_female,
        }])

        return df_combined

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all diseases and combine results
results = []
for disease_file in disease_files:
    disease_name = os.path.splitext(disease_file)[0]  # Remove .csv extension
    file_path = os.path.join(input_folder, disease_file)

    df_result = process_disease(file_path, disease_name)
    if df_result is not None:
        results.append(df_result)

# Merge all results into a single DataFrame
df_all = pd.concat(results, ignore_index=True)

manual_replacements = {
    "breast-cancer": "breast cancer",
    "chd": "chronic heart disease",
    "chronic-pain": "chronic pain",
    "copd": "chronic obstructive pulmonary disease",
    "current-asthma": "ongoing asthma",
}

# Apply replacements
df_all["disease"] = df_all["disease"].replace(manual_replacements)

# Keep a subset of columns
df_all = df_all[["disease", "percent_male", "percent_female"]]

# Save to CSV
df_all.to_csv(output_file, index=False)