
tasks_edu = [
    {
        "name": "Department of Education: Sex by Type of Degree",
        "dataset": "data/clean/edu.parquet",
        "variables": ["sex", "degree"],
        "prompt": "What is the sex of a person who completed a degree in {}?",
        "prompt_prob": "For a person who complete a degree in {} in the US, what is probability that they are {}?"
    },
]