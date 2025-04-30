tasks_meps = [
    {
        "name": "MEPS: Expenditure by Age Group, Q&A",
        "variables": ["expenditure", "age_group"],
        "prompt": "What is the annual healthcare expenditure for a person aged {} in the US? Answer with a number (in US dollars).",
        "levels": None,
    },
    {
        "name": "MEPS: Office-based Visits by Age Group, Q&A",
        "variables": ["office_visits", "age_group"],
        "prompt": "How many office-based medical visits does a person aged {} have per year? Answer with a number.",
        "levels": None,
    },
    {
        "name": "MEPS: Inpatient Visits by Age Group, Q&A",
        "variables": ["inpatient_visits", "age_group"],
        "prompt": "How many inpatient hospital visits does a person aged {} have per year? Answer with a number.",
        "levels": None,
    },
    {
        "name": "MEPS: Dental Visits by Age Group, Q&A",
        "variables": ["dental_visits", "age_group"],
        "prompt": "How many dental visits does a person aged {} have per year? Answer with a number.",
        "levels": None,
    },
    {
        "name": "MEPS: Has Insurance by Age Group, Q&A",
        "variables": ["insured", "age_group"],
        "prompt": "Does a person aged {} have any health insurance? Answer yes or no.",
        "levels": ["no", "yes"],
    },
    {
        "name": "MEPS: Expenditure by Race, Q&A",
        "variables": ["expenditure", "race"],
        "prompt": "What is the annual healthcare expenditure for a {} person in the US? Answer with a number (in US dollars).",
        "levels": None,
    },
    {
        "name": "MEPS: Office-based Visits by Race, Q&A",
        "variables": ["office_visits", "race"],
        "prompt": "How many office-based medical visits does a {} person have per year? Answer with a number.",
        "levels": None,
    },
    {
        "name": "MEPS: Inpatient Visits by Race, Q&A",
        "variables": ["inpatient_visits", "race"],
        "prompt": "How many inpatient hospital visits does a {} person have per year? Answer with a number.",
        "levels": None,
    },
    {
        "name": "MEPS: Dental Visits by Race, Q&A",
        "variables": ["dental_visits", "race"],
        "prompt": "How many dental visits does a {} person have per year? Answer with a number.",
        "levels": None,
    },
    {
        "name": "MEPS: Has Insurance by Race, Q&A",
        "variables": ["insured", "race"],
        "prompt": "Does a {} person have any health insurance? Answer yes or no.",
        "levels": ["no", "yes"],
    },
]

for t in tasks_meps:
    t["dataset"] = "data/clean/meps.parquet"