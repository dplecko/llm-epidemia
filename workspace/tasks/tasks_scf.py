
import sys
import os
sys.path.append(os.path.abspath("datasets"))
from helpers import bin_labels

tasks_scf = [
    # By age group
    {
        "name": "SCF: Food Expenditure by Age Group, Q&A",
        "variables": ["food_group", "age_group"],
        "prompt": "How much does a household spend on food if the primary respondent is aged {}?",
    },
    {
        "name": "SCF: House Ownership by Age Group, Q&A",
        "variables": ["house_own", "age_group"],
        "prompt": "Does a household where the primary respondent is aged {} own their home?",
    },
    {
        "name": "SCF: Total Assets by Age Group, Q&A",
        "variables": ["asset_group", "age_group"],
        "prompt": "What is the total value of assets for a household where the primary respondent is aged {}?",
    },
    {
        "name": "SCF: Debt by Age Group, Q&A",
        "variables": ["debt_group", "age_group"],
        "prompt": "What is the total debt of a household where the primary respondent is aged {}?",
    },
    {
        "name": "SCF: Net Worth by Age Group, Q&A",
        "variables": ["networth_group", "age_group"],
        "prompt": "What is the net worth of a household where the primary respondent is aged {}?",
    },
    # By race
    {
        "name": "SCF: Food Expenditure by Race, Q&A",
        "variables": ["food_group", "race"],
        "prompt": "How much does a household spend on food if the primary respondent is {}?",
    },
    {
        "name": "SCF: House Ownership by Race, Q&A",
        "variables": ["house_own", "race"],
        "prompt": "Does a household where the primary respondent is {} own their home?",
    },
    {
        "name": "SCF: Total Assets by Race, Q&A",
        "variables": ["asset_group", "race"],
        "prompt": "What is the total value of assets for a household where the primary respondent is {}?",
    },
    {
        "name": "SCF: Debt by Race, Q&A",
        "variables": ["debt_group", "race"],
        "prompt": "What is the total debt of a household where the primary respondent is {}?",
    },
    {
        "name": "SCF: Net Worth by Race, Q&A",
        "variables": ["networth_group", "race"],
        "prompt": "What is the net worth of a household where the primary respondent is {}?",
    },
    # By education
    {
        "name": "SCF: Food Expenditure by Education, Q&A",
        "variables": ["food_group", "education"],
        "prompt": "How much does a household spend on food if the primary respondent completed {}?",
    },
    {
        "name": "SCF: House Ownership by Education, Q&A",
        "variables": ["house_own", "education"],
        "prompt": "Does a household where the primary respondent completed {} own their home?",
    },
    {
        "name": "SCF: Total Assets by Education, Q&A",
        "variables": ["asset_group", "education"],
        "prompt": "What is the total value of assets for a household where the primary respondent completed {}?",
    },
    {
        "name": "SCF: Debt by Education, Q&A",
        "variables": ["debt_group", "education"],
        "prompt": "What is the total debt of a household where the primary respondent completed {}?",
    },
    {
        "name": "SCF: Net Worth by Education, Q&A",
        "variables": ["networth_group", "education"],
        "prompt": "What is the net worth of a household where the primary respondent completed {}?",
    }
]

for task in tasks_scf:
    task["dataset"] = "data/clean/scf.parquet"