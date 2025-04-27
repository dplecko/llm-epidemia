
tasks_gss = [
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