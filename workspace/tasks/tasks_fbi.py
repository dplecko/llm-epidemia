
tasks_fbi = [
    {
        "name": "FBI Crime Statistics: Sex by Crime Type, Q&A",
        "dataset": "data/clean/crime.parquet",
        "variables": ["sex", "crime_type"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person arrested for {}? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
]