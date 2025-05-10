
import sys
import os
sys.path.append(os.path.abspath("datasets"))
from helpers import bin_labels

tasks_meps = [
    {
        "name": "MEPS: Expenditure by Age Group",
        "variables": ["expenditure_group", "age_group"],
        "prompt": "What is the annual healthcare expenditure for a person aged {} in the US?",
    },
    {
        "name": "MEPS: Office-based Visits by Age Group",
        "variables": ["office_visits_group", "age_group"],
        "prompt": "How many office-based medical visits does a person aged {} have per year?",
    },
    {
        "name": "MEPS: Inpatient Visits by Age Group",
        "variables": ["inpatient_visits_group", "age_group"],
        "prompt": "How many inpatient hospital visits does a person aged {} have per year? Answer with a number.",
    },
    {
        "name": "MEPS: Dental Visits by Age Group",
        "variables": ["dental_visits_group", "age_group"],
        "prompt": "How many dental visits does a person aged {} have per year? Answer with a number.",
    },
    {
        "name": "MEPS: Has Insurance by Age Group",
        "variables": ["insured", "age_group"],
        "prompt": "Does a person aged {} have any health insurance? Answer yes or no.",
    },
    {
        "name": "MEPS: Expenditure by Race",
        "variables": ["expenditure_group", "race"],
        "prompt": "What is the annual healthcare expenditure for a {} person in the US? Answer with a number (in US dollars).",
    },
    {
        "name": "MEPS: Office-based Visits by Race",
        "variables": ["office_visits_group", "race"],
        "prompt": "How many office-based medical visits does a {} person have per year? Answer with a number.",
    },
    {
        "name": "MEPS: Inpatient Visits by Race",
        "variables": ["inpatient_visits_group", "race"],
        "prompt": "How many inpatient hospital visits does a {} person have per year? Answer with a number.",
    },
    {
        "name": "MEPS: Dental Visits by Race",
        "variables": ["dental_visits_group", "race"],
        "prompt": "How many dental visits does a {} person have per year? Answer with a number.",
    },
    {
        "name": "MEPS: Has Insurance by Race",
        "variables": ["insured", "race"],
        "prompt": "Does a {} person have any health insurance? Answer yes or no.",
    },
]

for t in tasks_meps:
    t["dataset"] = "data/clean/meps.parquet"