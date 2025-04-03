
task_specs = [
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
        "name": "Census: Age, Q&A",
        "dataset": "data/clean/census.parquet",
        "variables": ["sex", "age"],
        "cond_range": [18, 79],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person living in the United States who is {} years old? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
    # v0 tasks: Labor Department, Education Department, FBI Statistics
    {
        "name": "Department of Labor: Sex by Occupation, Q&A",
        "dataset": "data/clean/labor.parquet",
        "variables": ["sex", "occupation"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person working as a {}? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
    {
        "name": "FBI Crime Statistics: Sex by Crime Type, Q&A",
        "dataset": "data/clean/crime.parquet",
        "variables": ["sex", "crime_type"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person arrested for {}? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
    {
        "name": "Department of Education: Sex by Type of Degree, Q&A",
        "dataset": "data/clean/edu.parquet",
        "variables": ["sex", "degree"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person who completed a degree in {}? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
    {
        "name": "Labor: Sex by Occupation, Q&A",
        "dataset": "data/clean/labor.parquet",
        "variables": ["sex", "age"],
        "mode": "sample",
        "wgh_col": "weight",
        "prompt": "What is the sex of a person working as a {}? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    },
    # Census tasks
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
    # NHANES tasks
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
    # GSS tasks
    {
        "name": "GSS: Political View by Education Degree, Q&A (Sampling)",
        "dataset": "data/clean/gss.parquet",
        "variables": ["view", "degree"],
        "mode": "sample",
        "wgh_col": "wgh",
        "prompt": "What is the political view of a person with a {} degree? Single word answer (liberal, conservative, or moderate).",
        "levels": [["liberal", "Liberal"], ["moderate", "Moderate"], ["conservative", "Conservative"]]
    },
    {
        "name": "GSS: Political View by Education Degree, Q&A",
        "dataset": "data/clean/gss.parquet",
        "variables": ["view", "degree"],
        "mode": "logits",
        "wgh_col": "wgh",
        "prompt": "What is the political view of a person with a {} degree? Single word answer (liberal, conservative, or moderate).",
        "levels": [["liberal", "Liberal"], ["moderate", "Moderate"], ["conservative", "Conservative"]]
    },
    {
        "name": "GSS: Political Party by Education Degree, Q&A (Sampling)",
        "dataset": "data/clean/gss.parquet",
        "variables": ["party", "degree"],
        "mode": "sample",
        "wgh_col": "wgh",
        "prompt": "What is the political party affiliation of a person with a {} degree? Single word answer (democrat, republican, or independent).",
        "levels": [["democrat", "Democrat"], ["independent", "Independent"], ["republican", "Republican"]]
    }
]