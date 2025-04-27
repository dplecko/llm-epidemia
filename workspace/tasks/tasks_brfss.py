
tasks_brfss = [
    {
        "name": "BRFSS: Exercise by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["exercise_monthly", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Does a person living in {} exercise every month? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: Diabetes by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["diabetes", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Has a person living in {} ever been told they have diabetes? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: High BP by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["high_bp", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Does a person living in {} have high blood pressure? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: Asthma by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["asthma", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Does a person living in {} have asthma? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: Cholesterol by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["cholesterol", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Does a person living in {} have high cholesterol? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: Visual Impairments by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["blind", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Does a person living in {} have significant visual impairments/blindness? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: Hearing Impairments by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["blind", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Does a person living in {} have significant hearing impairments/deafness? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: Heart Attack by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["heart_attack", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Has a person living in {} ever suffered a heart attack? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "BRFSS: Stroke by State, Q&A",
        "dataset": "data/clean/brfss.parquet",
        "variables": ["stroke", "state"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "Has a person living in {} ever suffered a stroke? Answer with a single word (yes/no).",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
]