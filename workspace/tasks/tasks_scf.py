
tasks_scf = [
    # By age group
    {
        "name": "SCF: Food Expenditure by Age Group, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["food", "age_group"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "How much does a household spend on food if the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: House Ownership by Age Group, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["house_own", "age_group"],
        "mode": "logits",
        "wgh_col": "weight",
        "prompt": "Does a household where the primary respondent is aged {} own their home? Answer yes or no.",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "SCF: Total Assets by Age Group, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["asset", "age_group"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the total value of assets for a household where the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Debt by Age Group, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["debt", "age_group"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the total debt of a household where the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Net Worth by Age Group, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["networth", "age_group"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the net worth of a household where the primary respondent is aged {}? Answer with a number (in US dollars).",
        "levels": None
    },
        # By race
    {
        "name": "SCF: Food Expenditure by Race, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["food", "race"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "How much does a household spend on food if the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: House Ownership by Race, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["house_own", "race"],
        "mode": "logits",
        "wgh_col": "weight",
        "prompt": "Does a household where the primary respondent is {} own their home? Answer yes or no.",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "SCF: Total Assets by Race, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["asset", "race"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the total value of assets for a household where the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Debt by Race, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["debt", "race"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the total debt of a household where the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Net Worth by Race, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["networth", "race"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the net worth of a household where the primary respondent is {}? Answer with a number (in US dollars).",
        "levels": None
    },
        # By education
    {
        "name": "SCF: Food Expenditure by Education, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["food", "education"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "How much does a household spend on food if the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: House Ownership by Education, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["house_own", "education"],
        "mode": "logits",
        "wgh_col": "weight",
        "prompt": "Does a household where the primary respondent completed {} own their home? Answer yes or no.",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "SCF: Total Assets by Education, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["asset", "education"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the total value of assets for a household where the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Debt by Education, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["debt", "education"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the total debt of a household where the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    },
    {
        "name": "SCF: Net Worth by Education, Q&A",
        "dataset": "data/clean/scf.parquet",
        "variables": ["networth", "education"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the net worth of a household where the primary respondent completed {}? Answer with a number (in US dollars).",
        "levels": None
    }
]