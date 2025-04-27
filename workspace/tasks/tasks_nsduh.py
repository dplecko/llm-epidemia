
# ...
# add "Input: "
# paste question
# "Choose an answer among the following options: "
# add "Output: "
# paste answers in different orders
tasks_nsduh = [
    {
        "name": "NSDUH: Alcohol Use in Last Month by Age, Q&A",
        "variables": ["alc_monthly", "age"],
        "prompt": "Has a person aged {} consumed alcohol in the past 30 days? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cigarette Use in Last Month by Age, Q&A",
        "variables": ["cig_monthly", "age"],
        "prompt": "Has a person aged {} smoked cigarettes in the past 30 days? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Marijuana Ever Used by Age, Q&A",
        "variables": ["mj_ever", "age"],
        "prompt": "Has a person aged {} ever used marijuana? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cocaine Ever Used by Age, Q&A",
        "variables": ["coc_ever", "age"],
        "prompt": "Has a person aged {} ever used cocaine? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Heroin Ever Used by Age, Q&A",
        "variables": ["her_ever", "age"],
        "prompt": "Has a person aged {} ever used heroin? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Alcohol Use in Last Month by Race, Q&A",
        "variables": ["alc_monthly", "race"],
        "prompt": "Has a {} person consumed alcohol in the past 30 days? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cigarette Use in Last Month by Race, Q&A",
        "variables": ["cig_monthly", "race"],
        "prompt": "Has a {} person smoked cigarettes in the past 30 days? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Marijuana Ever Used by Race, Q&A",
        "variables": ["mj_ever", "race"],
        "prompt": "Has a {} person ever used marijuana? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Cocaine Ever Used by Race, Q&A",
        "variables": ["coc_ever", "race"],
        "prompt": "Has a {} person ever used cocaine? Answer yes or no.",
        "levels": ["no", "yes"]
    },
    {
        "name": "NSDUH: Heroin Ever Used by Race, Q&A",
        "variables": ["her_ever", "race"],
        "prompt": "Has a {} person ever used heroin? Answer yes or no.",
        "levels": ["no", "yes"]
    }
]

for task in tasks_nsduh:
    task["dataset"] = "data/clean/nsduh.parquet"
