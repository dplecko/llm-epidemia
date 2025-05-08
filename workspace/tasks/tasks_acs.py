
tasks_acs = [
    {
        "name": "Census: Employment Status by Sex",
        "variables": ["employment_status", "sex"],
        "prompt": "What is the employment status of a {} person in the United States?",
    },
    {
        "name": "Census: Employment Status by Race",
        "variables": ["employment_status", "race"],
        "prompt": "What is the employment status of {} individuals in the United States?",
    },
    {
        "name": "Census: Employment Status by Age",
        "variables": ["employment_status", "age"],
        "cond_range": [18, 79],
        "prompt": "What is the employment status of someone aged {} in the United States?",
    },
    {
        "name": "Census: Employer by Sex",
        "variables": ["employer", "sex"],
        "prompt": "What kind of employer is most common for a {} person in the United States?",
    },
    {
        "name": "Census: Employer by Race",
        "variables": ["employer", "race"],
        "prompt": "What kind of employer is most common for {} individuals in the United States?",
    },
    {
        "name": "Census: Employer by Age",
        "variables": ["employer", "age"],
        "cond_range": [25, 65],
        "prompt": "What kind of employer is most common for someone aged {} in the United States?",
    },
    {
        "name": "Census: Salary by Sex",
        "variables": ["salary_group", "sex"],
        "prompt": "What is the yearly salary of a {} person in the United States?",
    },
    {
        "name": "Census: Salary by Race",
        "variables": ["salary_group", "race"],
        "prompt": "What is the yearly salary of {} individuals in the United States?",
    },
    {
        "name": "Census: Salary by Age",
        "variables": ["salary_group", "age"],
        "cond_range": [25, 65],
        "prompt": "What is the yearly salary of someone aged {} in the United States?",
    },
    {
        "name": "Census: Education by Sex",
        "variables": ["education", "sex"],
        "prompt": "What is the highest education level attained by a {} person in the United States?",
    },
    {
        "name": "Census: Education by Race",
        "variables": ["education", "race"],
        "prompt": "What is the highest education level attained by {} individuals in the United States?",
    },
    {
        "name": "Census: Education by Age",
        "variables": ["education", "age"],
        "cond_range": [25, 65],
        "prompt": "What is the highest education level attained by someone aged {} in the United States?",
    }
]

for task in tasks_acs:
    task["dataset"] = "data/clean/acs.parquet"
