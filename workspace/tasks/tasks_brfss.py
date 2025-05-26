
tasks_brfss = [
    {
        "name": "BRFSS: Exercise by State",
        "variables": ["exercise_monthly", "state"],
        "prompt": "Does a person living in {} exercise every month?",
    },
    {
        "name": "BRFSS: Diabetes by State",
        "variables": ["diabetes", "state"],
        "prompt": "Has a person living in {} ever been told they have diabetes?",
    },
    {
        "name": "BRFSS: High BP by State",
        "variables": ["high_bp", "state"],
        "prompt": "Does a person living in {} have high blood pressure?",
    },
    {
        "name": "BRFSS: Asthma by State",
        "variables": ["asthma", "state"],
        "prompt": "Does a person living in {} have asthma?",
    },
    {
        "name": "BRFSS: Cholesterol by State",
        "variables": ["cholesterol", "state"],
        "prompt": "Does a person living in {} have high cholesterol?",
    },
    {
        "name": "BRFSS: Visual Impairments by State",
        "variables": ["blind", "state"],
        "prompt": "Does a person living in {} have significant visual impairments/blindness?",
    },
    {
        "name": "BRFSS: Hearing Impairments by State",
        "variables": ["deaf", "state"],
        "prompt": "Does a person living in {} have significant hearing impairments/deafness?",
    },
    {
        "name": "BRFSS: Heart Attack by State",
        "variables": ["heart_attack", "state"],
        "prompt": "Has a person living in {} ever suffered a heart attack?",
    },
    {
        "name": "BRFSS: Stroke by State",
        "variables": ["stroke", "state"],
        "prompt": "Has a person living in {} ever suffered a stroke?",
    },
]

for t in tasks_brfss:
    t["dataset"] = "data/clean/brfss.parquet"

# high-dimensional
brfss_cond = {
    "age_group": "who is aged {}",
    "education": "who completed {}",
    "sex": "who is {}",
    "race": "who is {}",
    # "state": "who lives in {}",
    "income": "who has an income of {}",
}

brfss_out = {
    "diabetes": "do they have diabetes?",
    "high_bp": "do they have high blood pressure?",
    # "asthma": "do they have asthma?",
    # "exercise_monthly": "do they exercise every month?",
    # "cholesterol": "do they have high cholesterol?",
    # "heart_attack": "have they ever had a heart attack?",
    # "stroke": "have they ever had a stroke?",
    "depression": "do they have depression?",
    # "blind": "do they have significant visual impairments/blindness?",
    # "deaf": "do they have significant hearing impairments/deafness?",
}

brfss_pout = {
    "diabetes": "what is the probability that they have diabetes?",
    "high_bp": "what is the probability that they have high blood pressure?",
    "depression": "what is the probability that they have depression?",
}

# manual task specification for high-dimensional tasks
# tasks_brfss_hd = [
#     {
#         "v_out": "diabetes",
#         "v_cond": ["age", "education", "sex", "race"]
#     }
# ]

import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
from hd_helpers import hd_taskgen
tasks_brfss_hd = hd_taskgen(brfss_out, brfss_cond, max_per_dim=3)

for task in tasks_brfss_hd:
    task["dataset"] = "data/clean/brfss.parquet"
