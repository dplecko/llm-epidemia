from datasets import load_dataset, logging
import json
logging.set_verbosity_error()


ds = load_dataset(
        "/Users/patrik/Documents/PhD/repos/llm-epidemia/workspace",   # ← full path to the script
        name="meta",                           # mandatory for local multi‑config
        trust_remote_code=True)
print(ds["meta"][0])                 # ['meta', 'data']
# print(ds["meta"][45]) 
# print(ds["acs.parquet"][2])  # dict

ds = load_dataset(
        "/Users/patrik/Documents/PhD/repos/llm-epidemia/workspace",   # ← full path to the script
        name="acs.parquet",                           # mandatory for local multi‑config
        trust_remote_code=True)
print(ds["data"][0])  