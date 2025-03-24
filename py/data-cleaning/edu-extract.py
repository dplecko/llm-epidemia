
# source: https://nces.ed.gov/ipeds/SummaryTables/report/360?templateId=3601&year=2023&expand_by=1&tt=aggregate&instType=1&sid=5ed94380-cc4e-494e-aae3-3293fcdde3c4
import pandas as pd
import os

file_path = "data/raw/education/edu.csv"
output_file = "data/clean/edu.csv"

df = pd.read_csv(file_path)
df['Men'] = pd.to_numeric(df['Men'].str.replace(',', ''), errors='coerce')
df['Women'] = pd.to_numeric(df['Women'].str.replace(',', ''), errors='coerce')

# compute the percentage of male graduates in each degree
df['percent_male'] = 100 * df['Men'] / (df['Men'] + df['Women'])
df['percent_female'] = 100 - df['percent_male']

df['degree'] = df['CIP Title'].str.lower().str.strip()

df[['degree', 'percent_male', 'percent_female']].to_csv(output_file, index=False)