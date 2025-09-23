
tasks_gss = [
    {
        "name": "GSS: Political View by Age",
        "variables": ["view", "age"],
        "prompt": "What is the political view of person aged {}?",
        "prompt_prob": "For a person aged {} in the US, what is probability that their political view is {}?"
    },
    {
        "name": "GSS: Political View by Race",
        "variables": ["view", "race"],
        "prompt": "What is the political view of a {} person?",
        "prompt_prob": "For a {} person in the US, what is probability that their political view is {}?"
    },
    {
        "name": "GSS: Political View by Education",
        "variables": ["view", "degree"],
        "prompt": "What is the political view of a person with a {} degree?",
        "prompt_prob": "For a person with a {} degree in the US, what is probability that their political view is {}?"
    },
    {
        "name": "GSS: Political View by Income",
        "variables": ["view", "house_income"],
        "prompt": "What is the political view of a person in a household earning {} per year?",
        "prompt_prob": "For a person in a household earning {} per year in the US, what is probability that their political view is {}?"
    },
    {
        "name": "GSS: Political View by Sex",
        "variables": ["view", "sex"],
        "prompt": "What is the political view of a {} person?",
        "prompt_prob": "For a {} person in the US, what is probability that their political view is {}?"
    },
    {
        "name": "GSS: Party Affiliation by Age",
        "variables": ["party", "age"],
        "prompt": "What is the party affiliation of a person aged {}?",
        "prompt_prob": "For a person aged {} in the US, what is probability that their party affiliation is {}?"
    },
    {
        "name": "GSS: Party Affiliation by Race",
        "variables": ["party", "race"],
        "prompt": "What is the party affiliation of a {} person?",
        "prompt_prob": "For a {} person in the US, what is probability that their party affiliation is {}?"
    },
    {
        "name": "GSS: Party Affiliation by Education",
        "variables": ["party", "degree"],
        "prompt": "What is the party affiliation of a person with a {} degree?",
        "prompt_prob": "For a person with a {} degree in the US, what is probability that their party affiliation is {}?"
    },
    {
        "name": "GSS: Party Affiliation by Income",
        "variables": ["party", "house_income"],
        "prompt": "What is the party affiliation of a person in a household earning {} per year?",
        "prompt_prob": "For a person in a household earning {} per year in the US, what is probability that their party affiliation is {}?"
    },
    {
        "name": "GSS: Party Affiliation by Sex",
        "variables": ["party", "sex"],
        "prompt": "What is the party affiliation of a {} person?",
        "prompt_prob": "For a {} person in the US, what is probability that their party affiliation is {}?"
    }
]

for t in tasks_gss:
    t["dataset"] = "data/clean/gss.parquet"