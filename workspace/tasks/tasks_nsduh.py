
tasks_nsduh = [
    {
        "name": "NSDUH: Alcohol Use in Last Month by Age",
        "variables": ["alc_monthly", "age"],
        "prompt": "Has a person aged {} consumed alcohol in the past 30 days?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cigarette Use in Last Month by Age",
        "variables": ["cig_monthly", "age"],
        "prompt": "Has a person aged {} smoked cigarettes in the past 30 days?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Marijuana Ever Used by Age",
        "variables": ["mj_ever", "age"],
        "prompt": "Has a person aged {} ever used marijuana?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cocaine Ever Used by Age",
        "variables": ["coc_ever", "age"],
        "prompt": "Has a person aged {} ever used cocaine?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Heroin Ever Used by Age",
        "variables": ["her_ever", "age"],
        "prompt": "Has a person aged {} ever used heroin?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Alcohol Use in Last Month by Race",
        "variables": ["alc_monthly", "race"],
        "prompt": "Has a {} person consumed alcohol in the past 30 days?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cigarette Use in Last Month by Race",
        "variables": ["cig_monthly", "race"],
        "prompt": "Has a {} person smoked cigarettes in the past 30 days?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Marijuana Ever Used by Race",
        "variables": ["mj_ever", "race"],
        "prompt": "Has a {} person ever used marijuana?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cocaine Ever Used by Race",
        "variables": ["coc_ever", "race"],
        "prompt": "Has a {} person ever used cocaine?",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Heroin Ever Used by Race",
        "variables": ["her_ever", "race"],
        "prompt": "Has a {} person ever used heroin?",
        "levels": ["no", "yes"]
    }
]

for task in tasks_nsduh:
    task["dataset"] = "data/clean/nsduh.parquet"
