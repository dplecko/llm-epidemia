
import sys
import os
sys.path.append(os.path.abspath("datasets"))
from helpers import bin_labels

tasks_scf = [
    # By age group
    {
        "name": "SCF: Food Expenditure by Age Group",
        "variables": ["food_group", "age_group"],
        "prompt": "How much does a household spend on food annually if the primary respondent is aged {}?",
        "prompt_prob": "For a household with primary respondent aged {}, in the US, what is probability that they spend {} on food annually?"
    },
    {
        "name": "SCF: House Ownership by Age Group",
        "variables": ["house_own", "age_group"],
        "prompt": "Does a household where the primary respondent is aged {} own their home?",
        "prompt_prob": "For a household with primary respondent aged {}, in the US, what is probability that they answer {} when asked if they own their home?"
    },
    {
        "name": "SCF: Total Assets by Age Group",
        "variables": ["asset_group", "age_group"],
        "prompt": "What is the total value of assets for a household where the primary respondent is aged {}?",
        "prompt_prob": "For a household with primary respondent aged {}, in the US, what is probability that their total value of assets is {}"
    },
    {
        "name": "SCF: Debt by Age Group",
        "variables": ["debt_group", "age_group"],
        "prompt": "What is the total debt of a household where the primary respondent is aged {}?",
        "prompt_prob": "For a household with primary respondent aged {}, in the US, what is probability that their total debt is {}"
    },
    {
        "name": "SCF: Net Worth by Age Group",
        "variables": ["networth_group", "age_group"],
        "prompt": "What is the net worth of a household where the primary respondent is aged {}?",
        "prompt_prob": "For a household with primary respondent aged {}, in the US, what is probability that their net worth is {}"
    },
    # By race
    {
        "name": "SCF: Food Expenditure by Race",
        "variables": ["food_group", "race"],
        "prompt": "How much does a household spend on food if the primary respondent is {}?",
        "prompt_prob": "For a household where the primary respondent is {}, in the US, what is probability that they spend {} on food annually?"
    },
    {
        "name": "SCF: House Ownership by Race",
        "variables": ["house_own", "race"],
        "prompt": "Does a household where the primary respondent is {} own their home?",
        "prompt_prob": "For a household where the primary respondent is {}, in the US, what is probability that they answer {} when asked if they own their home?"
    },
    {
        "name": "SCF: Total Assets by Race",
        "variables": ["asset_group", "race"],
        "prompt": "What is the total value of assets for a household where the primary respondent is {}?",
        "prompt_prob": "For a household where the primary respondent is {}, in the US, what is probability that their total value of assets is {}"
    },
    {
        "name": "SCF: Debt by Race",
        "variables": ["debt_group", "race"],
        "prompt": "What is the total debt of a household where the primary respondent is {}?",
        "prompt_prob": "For a household where the primary respondent is {}, in the US, what is probability that their total debt is {}"
    },
    {
        "name": "SCF: Net Worth by Race",
        "variables": ["networth_group", "race"],
        "prompt": "What is the net worth of a household where the primary respondent is {}?",
        "prompt_prob": "For a household where the primary respondent is {}, in the US, what is probability that their net worth is {}"
    },
    # By education
    {
        "name": "SCF: Food Expenditure by Education",
        "variables": ["food_group", "education"],
        "prompt": "How much does a household spend on food if the primary respondent completed {}?",
        "prompt_prob": "For a household where primary respondent completed {}, in the US, what is probability that they spend {} on food annually?"
    },
    {
        "name": "SCF: House Ownership by Education",
        "variables": ["house_own", "education"],
        "prompt": "Does a household where the primary respondent completed {} own their home?",
        "prompt_prob": "For a household where primary respondent completed {}, in the US, what is probability that they answer {} when asked if they own their home?"
    },
    {
        "name": "SCF: Total Assets by Education",
        "variables": ["asset_group", "education"],
        "prompt": "What is the total value of assets for a household where the primary respondent completed {}?",
        "prompt_prob": "For a household where primary respondent completed {}, in the US, what is probability that their total value of assets is {}"
    },
    {
        "name": "SCF: Debt by Education",
        "variables": ["debt_group", "education"],
        "prompt": "What is the total debt of a household where the primary respondent completed {}?",
        "prompt_prob": "For a household where primary respondent completed {}, in the US, what is probability that their total debt is {}"
    },
    {
        "name": "SCF: Net Worth by Education",
        "variables": ["networth_group", "education"],
        "prompt": "What is the net worth of a household where the primary respondent completed {}?",
        "prompt_prob": "For a household where primary respondent completed {}, in the US, what is probability that their net worth is {}"
    }
]

for task in tasks_scf:
    task["dataset"] = "data/clean/scf.parquet"

# high-dimensional
scf_cond = {
    "age_group": "who is {} years of age",
    "education": "who completed {}",
    "sex": "who is {}",
    "race": "who is {}",
}

scf_out = {
    "house_own": "do they own the home they live in?",
}

scf_pout = {
    "house_own": "what is the probability that they own the home they live in?",
}

# tasks_scf_hd = [
#     {
#         "v_out": "house_own",
#         "v_cond": ["age_group", "education", "sex", "race"]
#     }
# ]

import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
from hd_helpers import hd_taskgen
tasks_scf_hd = hd_taskgen(scf_out, scf_cond)

for task in tasks_scf_hd:
    task["dataset"] = "data/clean/scf.parquet"