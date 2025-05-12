
tasks_nsduh = [
    {
        "name": "NSDUH: Alcohol Use in Last Month by Age",
        "variables": ["alc_monthly", "age"],
        "prompt": "Has a person aged {} consumed alcohol in the past 30 days?",
    },
    {
        "name": "NSDUH: Cigarette Use in Last Month by Age",
        "variables": ["cig_monthly", "age"],
        "prompt": "Has a person aged {} smoked cigarettes in the past 30 days?",
    },
    {
        "name": "NSDUH: Marijuana Ever Used by Age",
        "variables": ["mj_ever", "age"],
        "prompt": "Has a person aged {} ever used marijuana?",
    },
    {
        "name": "NSDUH: Cocaine Ever Used by Age",
        "variables": ["coc_ever", "age"],
        "prompt": "Has a person aged {} ever used cocaine?",
    },
    {
        "name": "NSDUH: Heroin Ever Used by Age",
        "variables": ["her_ever", "age"],
        "prompt": "Has a person aged {} ever used heroin?",
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
    "alc_monthly": "have they drank alcohol in the past 30 days?",
    "mj_ever": "have they ever used marijuana?",
    "coc_ever": "have they ever used cocaine?",
    "her_ever": "have they ever used heroin?",
}

# tasks_nsduh_hd = [
#     {
#         "v_out": "mj_ever",
#         "v_cond": ["age", "edu", "sex", "race"]
#     }
# ]

import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
from helpers import hd_taskgen
tasks_nsduh_hd = hd_taskgen(nsduh_out, nsduh_cond)

for task in tasks_nsduh_hd:
    task["dataset"] = "data/clean/nsduh.parquet"