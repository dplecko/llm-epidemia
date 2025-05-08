
tasks_labor = [
    {
        "name": "Department of Labor: Sex by Occupation",
        "dataset": "data/clean/labor.parquet",
        "variables": ["sex", "occupation"],
        "prompt": "What is the sex of a person working as a {}?",
    },
    {
        "name": "Department of Labor: Race by Occupation",
        "dataset": "data/clean/labor.parquet",
        "variables": ["race", "occupation"],
        "prompt": "What is the race of a person working as a {}?",
    },
]