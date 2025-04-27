
tasks_edu = [
    {
        "name": "Department of Education: Sex by Type of Degree, Q&A",
        "dataset": "data/clean/edu.parquet",
        "variables": ["sex", "degree"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person who completed a degree in {}? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
]