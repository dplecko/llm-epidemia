
tasks_nsduh = [
    {
        "name": "NSDUH: Alcohol Use in Last Month by Age",
        "variables": ["alc_monthly", "age"],
        "prompt": "Has a person aged {} consumed alcohol in the past 30 days?",
        "prompt_prob": "For a person aged {} in the US, what is probability that they answer {} when asked if they drink alcohol monthly?"
    },
    {
        "name": "NSDUH: Cigarette Use in Last Month by Age",
        "variables": ["cig_monthly", "age"],
        "prompt": "Has a person aged {} smoked cigarettes in the past 30 days?",
        "prompt_prob": "For a person aged {} in the US, what is probability that they answer {} when asked if they smoke cigarettes monthly?"
    },
    {
        "name": "NSDUH: Marijuana Ever Used by Age",
        "variables": ["mj_ever", "age"],
        "prompt": "Has a person aged {} ever used marijuana?",
        "prompt_prob": "For a person aged {} in the US, what is probability that they answer {} when asked if they ever used marijuana?"
    },
    {
        "name": "NSDUH: Cocaine Ever Used by Age",
        "variables": ["coc_ever", "age"],
        "prompt": "Has a person aged {} ever used cocaine?",
        "prompt_prob": "For a person aged {} in the US, what is probability that they answer {} when asked if they ever used cocaine?"
    },
    {
        "name": "NSDUH: Heroin Ever Used by Age",
        "variables": ["her_ever", "age"],
        "prompt": "Has a person aged {} ever used heroin?",
        "prompt_prob": "For a person aged {} in the US, what is probability that they answer {} when asked if they ever used heroin?"
    },
    {
        "name": "NSDUH: Alcohol Use in Last Month by Race",
        "variables": ["alc_monthly", "race"],
        "prompt": "Has a {} person consumed alcohol in the past 30 days?",
    },
    {
        "name": "NSDUH: Cigarette Use in Last Month by Race",
        "variables": ["cig_monthly", "race"],
        "prompt": "Has a {} person smoked cigarettes in the past 30 days?",
    },
    {
        "name": "NSDUH: Marijuana Ever Used by Race",
        "variables": ["mj_ever", "race"],
        "prompt": "Has a {} person ever used marijuana?",
    },
    {
        "name": "NSDUH: Cocaine Ever Used by Race",
        "variables": ["coc_ever", "race"],
        "prompt": "Has a {} person ever used cocaine?",
    },
    {
        "name": "NSDUH: Heroin Ever Used by Race",
        "variables": ["her_ever", "race"],
        "prompt": "Has a {} person ever used heroin?",
    }
]

for task in tasks_nsduh:
    task["dataset"] = "data/clean/nsduh.parquet"

# high-dimensional
nsduh_cond = {
    "age": "who is {} years of age",
    "edu": "who completed {}",
    "sex": "who is {}",
    "race": "who is {}",
}

nsduh_out = {
    "cig_monthly": "have they smoked in the past 30 days?",
    # "alc_monthly": "have they drank alcohol in the past 30 days?",
    "mj_ever": "have they ever used marijuana?",
    "coc_ever": "have they ever used cocaine?",
    # "her_ever": "have they ever used heroin?",
}

nsduh_pout = {
    "cig_monthly": "what is the probability that they smoked in the past 30 days?",
    "mj_ever": "what is the probability that they ever used marijuana?",
    "coc_ever": "what is the probability that they ever used cocaine?",
}

# tasks_nsduh_hd = [
#     {
#         "v_out": "mj_ever",
#         "v_cond": ["age", "edu", "sex", "race"]
#     }
# ]

import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
from hd_helpers import hd_taskgen
tasks_nsduh_hd = hd_taskgen(nsduh_out, nsduh_cond)

for task in tasks_nsduh_hd:
    task["dataset"] = "data/clean/nsduh.parquet"