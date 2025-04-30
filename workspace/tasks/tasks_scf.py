tasks_scf = [
    # By age group
    {
        "name": "SCF: Food Expenditure by Age Group, Q&A",
        "variables": ["food", "age_group"],
        "prompt": "How much does a household spend on food if the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: House Ownership by Age Group, Q&A",
        "variables": ["house_own", "age_group"],
        "prompt": "Does a household where the primary respondent is aged {} own their home? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "SCF: Total Assets by Age Group, Q&A",
        "variables": ["asset", "age_group"],
        "prompt": "What is the total value of assets for a household where the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Debt by Age Group, Q&A",
        "variables": ["debt", "age_group"],
        "prompt": "What is the total debt of a household where the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Net Worth by Age Group, Q&A",
        "variables": ["networth", "age_group"],
        "prompt": "What is the net worth of a household where the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
        # By race
    {
        "name": "SCF: Food Expenditure by Race, Q&A",
        "variables": ["food", "race"],
        "prompt": "How much does a household spend on food if the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: House Ownership by Race, Q&A",
        "variables": ["house_own", "race"],
        "prompt": "Does a household where the primary respondent is {} own their home? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "SCF: Total Assets by Race, Q&A",
        "variables": ["asset", "race"],
        "prompt": "What is the total value of assets for a household where the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Debt by Race, Q&A",
        "variables": ["debt", "race"],
        "prompt": "What is the total debt of a household where the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Net Worth by Race, Q&A",
        "variables": ["networth", "race"],
        "prompt": "What is the net worth of a household where the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
        # By education
    {
        "name": "SCF: Food Expenditure by Education, Q&A",
        "variables": ["food", "education"],
        "prompt": "How much does a household spend on food if the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: House Ownership by Education, Q&A",
        "variables": ["house_own", "education"],
        "prompt": "Does a household where the primary respondent completed {} own their home? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "SCF: Total Assets by Education, Q&A",
        "variables": ["asset", "education"],
        "prompt": "What is the total value of assets for a household where the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Debt by Education, Q&A",
        "variables": ["debt", "education"],
        "prompt": "What is the total debt of a household where the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Net Worth by Education, Q&A",
        "variables": ["networth", "education"],
        "prompt": "What is the net worth of a household where the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    }
]

for task in tasks_scf:
    task["dataset"] = "data/clean/scf.parquet"