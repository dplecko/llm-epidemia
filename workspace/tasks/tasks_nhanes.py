
tasks_nhanes = [
    {
        "name": "NHANES: Age by BMI Group",
        "variables": ["age_group", "bmi_bin"],
        "prompt": "What is the age of a person with body mass index (BMI) in the {} range?",
    },
    {
        "name": "NHANES: Diabetes by BMI Group",
        "variables": ["diabetes", "bmi_bin"],
        "prompt": "Does a person with a body mass index (BMI) in the {} range have diabetes?",
    },
    {
        "name": "NHANES: Diabetes by Age Group",
        "variables": ["diabetes", "age_group"],
        "prompt": "Does a person aged {} have diabetes?",
    },
    {
        "name": "NHANES: Weekly Alcohol Consumption by Age Group",
        "variables": ["alcohol_weekly", "age_group"],
        "prompt": "Does a person in age group {} drink alcohol weekly?",
    },
]

for t in tasks_nhanes:
    t["dataset"] = "data/clean/nhanes.parquet"