
---
license: apache-2.0
---


<div align="center">
  <h1 style="display:inline-flex; align-items:center; gap:0.5rem; white-space:nowrap; margin:0;">
    <img
      src="https://huggingface.co/datasets/llm-observatory/llm-observatory/resolve/main/img/logo-icon.png"
      alt="LLM Observatory"
      width="55"
      height="40"
      style="display:block;"
    />
    <span style="font-size:2em; margin:0;">LLM Observatory</span>
  </h1>
</div>



<p align="center">
  <a href="https://llm-observatory.org/index.html">
    <img src="https://img.shields.io/badge/Website-Visit-blue?style=flat" alt="Website">
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/dplecko/llm-epidemia">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=flat&logo=github" alt="GitHub">
  </a>
</p>


This repository holds the data used by the [LLM Observatory](https://llm-observatory.org/index.html), an iniative for monitoring, understanding, and mapping probabilistic capabilities of Large Language Models.
For benchmarks of the LLM Observatory, check the [LLM Observatory Workspace](https://huggingface.co/spaces/llm-observatory/llm-observatory-eval).



## Datasets

Preprocessed files for the following datasets are available, grouped into domains:

### Health

- [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_2023.html)  
- [National Health and Nutrition Examination Survey (NHANES)](https://www.cdc.gov/nchs/nhanes/about/index.html)  
- [National Survey on Drug Use and Health (NSDUH)](https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/national-releases/2023)  
- [Medical Expenditure Panel Survey (MEPS)](https://meps.ahrq.gov/)  

### Social

- [General Social Survey (GSS)](https://gss.norc.org/)  
- [FBI Arrest Data (FBI Arrests)](https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/tables/table-42/table-42.xls)  
- [IPEDS Education Data (IPEDS)](https://nces.ed.gov/ipeds/)  

### Economic

- [Bureau of Labor Statistics (BLS)](https://www.bls.gov/)  
- [American Community Survey (ACS)](https://www.census.gov/programs-surveys/acs.html)  
- [Survey of Consumer Finances (SCF)](https://www.federalreserve.gov/econres/scfindex.htm)  


## Getting Started
You can load one of the datasets with:
```python
load_dataset("llm-observatory/llm-observatory", "<dataset>", split="train", trust_remote_code=True).to_pandas()
```
or by downloading the file:
```python
hf_hub_download(
    repo_id="llm-observatory/llm-observatory", 
    filename=f"data/<dataset>.parquet", 
    repo_type="dataset"
)
```


##  Citation Information
```
@techreport{llmobservatory2025layer1,
  title={Epidemiology of Large Language Models: A Benchmark for Observational Distribution Knowledge},
  author={Plecko, Drago and Okanovic, Patrik and Hoeffler, Torsten and Bareinboim, Elias},
  institution={Causal AI Lab, Columbia University},
  year={2025},
  url={\url{https://causalai.net/r136.pdf}}
}
```

## Contribute
LLM Observatory is an open-source initiative interested in your contributions.
Further details on contributing can be found at this link: [contribute](https://llm-observatory.org/contribute.html).