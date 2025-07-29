
tasks_nhanes = [
    {
        "name": "NHANES: Age by BMI Group",
        "variables": ["age_group", "bmi_bin"],
        "prompt": "What is the age of a person with body mass index (BMI) in the {} range?",
        "prompt_prob": "For a person with BMI {} in the US, what is probability that their age is {}?"
    },
    {
        "name": "NHANES: Diabetes by BMI Group",
        "variables": ["diabetes", "bmi_bin"],
        "prompt": "Does a person with a body mass index (BMI) in the {} range have diabetes?",
        "prompt_prob": "For a person with BMI {} in the US, what is probability that they answer {} when asked if they have diabetes?"
    },
    {
        "name": "NHANES: Diabetes by Age Group",
        "variables": ["diabetes", "age_group"],
        "prompt": "Does a person aged {} have diabetes?",
        "prompt_prob": "For a person aged {} in the US, what is probability that they answer {} when asked if they have diabetes?"
    },
    {
        "name": "NHANES: Weekly Alcohol Consumption by Age Group",
        "variables": ["alcohol_weekly", "age_group"],
        "prompt": "Does a person in age group {} drink alcohol weekly?",
        "prompt_prob": "For a person aged {} in the US, what is probability that they answer {} when asked if they drink alcohol weekly?"
    },
]

for t in tasks_nhanes:
    t["dataset"] = "data/clean/nhanes.parquet"