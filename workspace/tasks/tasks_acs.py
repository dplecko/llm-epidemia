tasks_acs = [
    {
        "name": "Census: Employment Status by Sex",
        "variables": ["employment_status", "sex"],
        "prompt": "What is the employment status of a {} person in the United States?",
        "levels": ["employed", "not at work", "unemployed", "not in labor force"]
    },
    {
        "name": "Census: Employment Status by Race",
        "variables": ["employment_status", "race"],
        "prompt": "What is the employment status of {} individuals in the United States?",
        "levels": ["employed", "not at work", "unemployed", "not in labor force"]
    },
    {
        "name": "Census: Employment Status by Age",
        "variables": ["employment_status", "age"],
        "cond_range": [18, 79],
        "prompt": "What is the employment status of someone aged {} in the United States?",
        "levels": ["employed", "not at work", "unemployed", "not in labor force"]
    },
    {
        "name": "Census: Salary by Sex",
        "variables": ["salary", "sex"],
        "prompt": "What is the average salary of a {} person in the United States?",
        "levels": "continuous"
    },
    {
        "name": "Census: Salary by Race",
        "variables": ["salary", "race"],
        "prompt": "What is the average salary of {} individuals in the United States?",
        "levels": "continuous"
    },
    {
        "name": "Census: Salary by Age",
        "variables": ["salary", "age"],
        "cond_range": [25, 65],
        "prompt": "What is the average salary of someone aged {} in the United States?",
        "levels": "continuous"
    },
    {
        "name": "Census: Employer by Sex",
        "variables": ["employer", "sex"],
        "prompt": "What kind of employer is most common for a {} person in the United States?",
        "levels": ["for-profit company", "non-profit company", "government", "self-employed"]
    },
    {
        "name": "Census: Employer by Race",
        "variables": ["employer", "race"],
        "prompt": "What kind of employer is most common for {} individuals in the United States?",
        "levels": ["for-profit company", "non-profit company", "government", "self-employed"]
    },
    {
        "name": "Census: Employer by Age",
        "variables": ["employer", "age"],
        "cond_range": [25, 65],
        "prompt": "What kind of employer is most common for someone aged {} in the United States?",
        "levels": ["for-profit company", "non-profit company", "government", "self-employed"]
    },
    {
        "name": "Census: Education by Sex",
        "variables": ["education", "sex"],
        "prompt": "What is the highest education level typically attained by a {} person in the United States?",
        "levels": [
            "No schooling completed", "Nursery school, preschool", "Kindergarten",
            "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6",
            "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "12th grade - no diploma",
            "Regular high school diploma", "GED or alternative credential",
            "Some college, but less than 1 year",
            "1 or more years of college credit, no degree", "Associate's degree",
            "Bachelor's degree", "Master's degree",
            "Professional degree beyond a bachelor's degree", "Doctorate degree"
        ]
    },
    {
        "name": "Census: Education by Race",
        "variables": ["education", "race"],
        "prompt": "What is the highest education level typically attained by {} individuals in the United States?",
        "levels": [
            "No schooling completed", "Nursery school, preschool", "Kindergarten",
            "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6",
            "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "12th grade - no diploma",
            "Regular high school diploma", "GED or alternative credential",
            "Some college, but less than 1 year",
            "1 or more years of college credit, no degree", "Associate's degree",
            "Bachelor's degree", "Master's degree",
            "Professional degree beyond a bachelor's degree", "Doctorate degree"
        ]
    },
    {
        "name": "Census: Education by Age",
        "variables": ["education", "age"],
        "cond_range": [25, 65],
        "prompt": "What is the highest education level typically attained by someone aged {} in the United States?",
        "levels": [
            "No schooling completed", "Nursery school, preschool", "Kindergarten",
            "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6",
            "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "12th grade - no diploma",
            "Regular high school diploma", "GED or alternative credential",
            "Some college, but less than 1 year",
            "1 or more years of college credit, no degree", "Associate's degree",
            "Bachelor's degree", "Master's degree",
            "Professional degree beyond a bachelor's degree", "Doctorate degree"
        ]
    }
]
