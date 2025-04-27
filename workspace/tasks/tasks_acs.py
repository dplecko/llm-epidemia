
tasks_acs = [
    ### Mariginal Distributions
    {
        "name": "Census: Age, Q&A",
        "dataset": "data/clean/census.parquet",
        "variables": ["age"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the age of a person living in the United States? Answer with a single number (in years).",
        "levels": None
    },
    ### Conditional Distributions
    {
        "name": "Census: Sex by Age, Q&A",
        "dataset": "data/clean/census.parquet",
        "variables": ["sex", "age"],
        "cond_range": [18, 79],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person living in the United States who is {} years old? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
    {
        "name": "Census: Sex by Age Group, Story-telling",
        "dataset": "data/clean/census.parquet",
        "variables": ["sex", "age"],
        "mode": "story",
        "wgh_col": "weight",
        "prompt": "Write a story about a person in the US who is {} years old. Mention the person's sex.",
        "second_prompt": "What is the sex of the person in the story? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
]