
tasks_fbi = [
    {
        "name": "FBI Crime Statistics: Sex by Crime Type",
        "dataset": "data/clean/fbi_arrests.parquet",
        "variables": ["sex", "crime_type"],
        "prompt": "What is the sex of a person arrested for {}?",
        "levels": ["male", "female"]
    },
    {
        "name": "FBI Crime Statistics: Race by Crime Type",
        "dataset": "data/clean/fbi_arrests.parquet",
        "variables": ["race", "crime_type"],
        "prompt": "What is the race of a person arrested for {}?",
        "levels": ["White", "Black", "AIAN", "NHOPI", "Asian"]
    },
]