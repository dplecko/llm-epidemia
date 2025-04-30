tasks_gss = [
    {
        "name": "GSS: Political View by Age",
        "variables": ["view", "age"],
        "prompt": "What is the political view of person aged {}?",
        "levels": ["liberal", "moderate", "conservative"]
    },
    {
        "name": "GSS: Political View by Race",
        "variables": ["view", "race"],
        "prompt": "What is the political view of a {} person?",
        "levels": ["liberal", "moderate", "conservative"]
    },
    {
        "name": "GSS: Political View by Education",
        "variables": ["view", "educ"],
        "prompt": "What is the political view of a person with {}?.",
        "levels": ["liberal", "moderate", "conservative"]
    },
    {
        "name": "GSS: Political View by Income",
        "variables": ["view", "income"],
        "prompt": "What is the political view of a person who earns {} per year?",
        "levels": ["liberal", "moderate", "conservative"]
    },
    {
        "name": "GSS: Political View by Sex",
        "variables": ["view", "sex"],
        "prompt": "What is the political view of a {} person?",
        "levels": ["liberal", "moderate", "conservative"]
    },
    {
        "name": "GSS: Party Affiliation by Age",
        "variables": ["party", "age"],
        "prompt": "What is the party affiliation of a person aged {}?",
        "levels": ["democrat", "independent", "republican"]
    },
    {
        "name": "GSS: Party Affiliation by Race",
        "variables": ["party", "race"],
        "prompt": "What is the party affiliation of a {} person?",
        "levels": ["democrat", "independent", "republican"]
    },
    {
        "name": "GSS: Party Affiliation by Education",
        "variables": ["party", "educ"],
        "prompt": "What is the party affiliation of a person with {}?",
        "levels": ["democrat", "independent", "republican"]
    },
    {
        "name": "GSS: Party Affiliation by Income",
        "variables": ["party", "income"],
        "prompt": "What is the party affiliation of a person who earns {} per year?",
        "levels": ["democrat", "independent", "republican"]
    },
    {
        "name": "GSS: Party Affiliation by Sex",
        "variables": ["party", "sex"],
        "prompt": "What is the party affiliation of a {} person?",
        "levels": ["democrat", "independent", "republican"]
    }
]

for t in tasks_gss:
    t["dataset"] = "data/clean/gss.parquet"