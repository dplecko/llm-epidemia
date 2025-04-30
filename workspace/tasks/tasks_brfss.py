
tasks_brfss = [
    {
        "name": "BRFSS: Exercise by State",
        "variables": ["exercise_monthly", "state"],
        "prompt": "Does a person living in {} exercise every month?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Diabetes by State",
        "variables": ["diabetes", "state"],
        "prompt": "Has a person living in {} ever been told they have diabetes?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: High BP by State",
        "variables": ["high_bp", "state"],
        "prompt": "Does a person living in {} have high blood pressure?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Asthma by State",
        "variables": ["asthma", "state"],
        "prompt": "Does a person living in {} have asthma?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Cholesterol by State",
        "variables": ["cholesterol", "state"],
        "prompt": "Does a person living in {} have high cholesterol?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Visual Impairments by State",
        "variables": ["blind", "state"],
        "prompt": "Does a person living in {} have significant visual impairments/blindness?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Hearing Impairments by State",
        "variables": ["blind", "state"],
        "prompt": "Does a person living in {} have significant hearing impairments/deafness?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Heart Attack by State",
        "variables": ["heart_attack", "state"],
        "prompt": "Has a person living in {} ever suffered a heart attack?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Stroke by State",
        "variables": ["stroke", "state"],
        "prompt": "Has a person living in {} ever suffered a stroke?",
        "levels": ["no", "yes"]
    },
]

for t in tasks_brfss:
    t["dataset"] = "data/clean/brfss.parquet"