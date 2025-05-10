
tasks_acs = [
    {
        "name": "Census: Employment Status by Sex",
        "variables": ["employment_status", "sex"],
        "subset": ["age", "lwr", 16],
        "prompt": "What is the employment status of a {} person in the United States?",
    },
    {
        "name": "Census: Employment Status by Race",
        "variables": ["employment_status", "race"],
        "subset": ["age", "lwr", 16],
        "prompt": "What is the employment status of {} individuals in the United States?",
    },
    {
        "name": "Census: Employment Status by Age",
        "variables": ["employment_status", "age"],
        "subset": ["age", "lwr", 16],
        "cond_range": [18, 79],
        "prompt": "What is the employment status of someone aged {} in the United States?",
    },
    {
        "name": "Census: Employer by Sex",
        "variables": ["employer", "sex"],
        "subset": ["employment_status", "levels", ["employed", "not at work"]],
        "prompt": "Who is the employer of a {} person working in the United States?",
    },
    {
        "name": "Census: Employer by Race",
        "variables": ["employer", "race"],
        "subset": ["employment_status", "levels", ["employed", "not at work"]],
        "prompt": "Who is the employer of a {} person working in the United States?",
    },
    {
        "name": "Census: Employer by Age",
        "variables": ["employer", "age"],
        "subset": ["employment_status", "levels", ["employed", "not at work"]],
        "cond_range": [25, 65],
        "prompt": "Who is the employer of a person aged {} working in the United States?",
    },
    {
        "name": "Census: Salary by Sex",
        "variables": ["salary_group", "sex"],
        "subset": ["employment_status", "levels", ["employed", "not at work"]],
        "prompt": "What is the yearly salary of a {} person in the United States?",
    },
    {
        "name": "Census: Salary by Race",
        "variables": ["salary_group", "race"],
        "subset": ["employment_status", "levels", ["employed", "not at work"]],
        "prompt": "What is the yearly salary of a {} person  in the United States?",
    },
    {
        "name": "Census: Salary by Age",
        "variables": ["salary_group", "age"],
        "subset": ["employment_status", "levels", ["employed", "not at work"]],
        "cond_range": [25, 65],
        "prompt": "What is the yearly salary of a person aged {} in the United States?",
    },
    {
        "name": "Census: Education by Sex",
        "variables": ["education", "sex"],
        "prompt": "What is the highest education level attained by a {} person in the United States?",
    },
    {
        "name": "Census: Education by Race",
        "variables": ["education", "race"],
        "prompt": "What is the highest education level attained by a {} person in the United States?",
    },
    {
        "name": "Census: Education by Age",
        "variables": ["education", "age"],
        "cond_range": [25, 65],
        "prompt": "What is the highest education level attained by a person aged {} in the United States?",
    }
]

for task in tasks_acs:
    task["dataset"] = "data/clean/acs.parquet"
