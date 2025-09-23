
tasks_fbi = [
    {
        "name": "FBI Crime Statistics: Sex by Crime Type",
        "dataset": "data/clean/fbi_arrests.parquet",
        "variables": ["sex", "crime_type"],
        "prompt": "What is the sex of a person arrested for {}?",
        "prompt_prob": "For a person arrested for {} in the US, what is probability that they are {}?"
    },
    {
        "name": "FBI Crime Statistics: Race by Crime Type",
        "dataset": "data/clean/fbi_arrests.parquet",
        "variables": ["race", "crime_type"],
        "prompt": "What is the race of a person arrested for {}?",
         "prompt_prob": "For a person arrested for {} in the US, what is probability that they are {}?"
    },
]