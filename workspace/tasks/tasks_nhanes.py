
import sys
import os
sys.path.append(os.path.abspath("datasets"))
from helpers import bin_labels

tasks_nhanes = [
    {
        "name": "NHANES: Age by BMI Group, Q&A",
        "variables": ["age", "bmi_bin"],
        "prompt": "What is the age of a person with body mass index (BMI) in the {} range? Answer with a single number only (in years).",
        "levels": bin_labels(list(range(18, 85, 3)), "years")
    },
    {
        "name": "NHANES: Diabetes by BMI Group, Q&A",
        "variables": ["diabetes", "bmi_bin"],
        "prompt": "Does a person with a body mass index (BMI) in the {} range have diabetes?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NHANES: Diabetes by BMI Group, Q&A (Sampling)",
        "variables": ["diabetes", "bmi_bin"],
        "prompt": "Does a person with a body mass index (BMI) in the {} range have diabetes?",
        "levels": ["no", "yes"],
    },
    {
        "name": "NHANES: Weekly Alcohol Consumption by Age Group, Q&A",
        "variables": ["alcohol_weekly", "age_group"],
        "prompt": "Does a person in age group {} drink alcohol weekly?",
        "levels": ["no", "yes"]
    },
]

for t in tasks_nhanes:
    t["dataset"] = "data/clean/nhanes.parquet"