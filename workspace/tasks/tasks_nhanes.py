
tasks_nhanes = [
    {
        "name": "NHANES: Age by BMI Group, Q&A",
        "dataset": "data/clean/nhanes.parquet",
        "variables": ["age", "bmi_bin"],
        "mode": "sample",
        "wgh_col": "mec_wgh",
        "prompt": "What is the age of a person with body mass index (BMI) in the {} range? Answer with a single number only (in years).",
        "levels": None
    },
    {
        "name": "NHANES: Diabetes by BMI Group, Q&A",
        "dataset": "data/clean/nhanes.parquet",
        "variables": ["diabetes", "bmi_bin"],
        "mode": "logits",
        "wgh_col": "mec_wgh",
        "prompt": "Does a person with a body mass index (BMI) in the {} range have diabetes? Single word yes or no answer.",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
    {
        "name": "NHANES: Diabetes by BMI Group, Q&A (Sampling)",
        "dataset": "data/clean/nhanes.parquet",
        "variables": ["diabetes", "bmi_bin"],
        "mode": "sample",
        "wgh_col": "mec_wgh",
        "prompt": "Does a person with a body mass index (BMI) in the {} range have diabetes? Single word yes or no answer.",
        "levels": [["No", "no", "NO"], ["Yes", "yes", "YES"]]
    },
    {
        "name": "NHANES: Weekly Alcohol Consumption by Age Group, Q&A",
        "dataset": "data/clean/nhanes.parquet",
        "variables": ["alcohol_weekly", "age_group"],
        "mode": "logits",
        "wgh_col": "mec_wgh",
        "prompt": "Does a person in age group {} drink alcohol weekly? Single word yes or no answer.",
        "levels": [["No", "no"], ["Yes", "yes"]]
    },
]