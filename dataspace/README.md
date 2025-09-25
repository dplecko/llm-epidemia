
---
license: apache-2.0
---



This repository holds the data used by the LLM Observatory, an iniative for monitoring, understanding, and mapping probabilistic capabilities of Large Language Models.
For benchmarks of the LLM Observatory.



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
load_dataset("<id>", "<dataset>", split="train", trust_remote_code=True).to_pandas()
```
or by downloading the file:
```python
hf_hub_download(
    repo_id="<id>", 
    filename=f"data/<dataset>.parquet", 
    repo_type="dataset"
)
```

