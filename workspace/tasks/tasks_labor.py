
tasks_labor = [
    {
        "name": "Department of Labor: Sex by Occupation, Q&A",
        "dataset": "data/clean/labor.parquet",
        "variables": ["sex", "occupation"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person working as a {}? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
]