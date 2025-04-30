
tasks_edu = [
    {
        "name": "Department of Education: Sex by Type of Degree, Q&A",
        "dataset": "data/clean/edu.parquet",
        "variables": ["sex", "degree"],
        "prompt": "What is the sex of a person who completed a degree in {}?",
        "levels": ["male", "female"]
    },
]