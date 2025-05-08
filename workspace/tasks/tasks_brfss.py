
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
        "variables": ["blind", "state"],
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