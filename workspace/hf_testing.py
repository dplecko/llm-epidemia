
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import os
import evaluate
# from workspace.common import *
# from task_spec import task_specs

# login(os.environ["HF_TOKEN"])
# df = load_dataset("llm-observatory/llm-observatory", "meps", split="train", trust_remote_code=True).to_pandas()

# df_lst = []
# dts = ["brfss", "nhanes", "nsduh", "gss", "fbi_arrests", "edu", "labor", "scf", "acs"]
# for i in range(len(dts)):
#     df_lst.append(load_dataset("llm-observatory/llm-observatory", dts[i], split="train", trust_remote_code=True).to_pandas())

# def load_local(dataset):
#     return pd.read_parquet(f"data/clean/{dataset}.parquet")

# for i in range(len(dts)):
#     print(dts[i], " ", df_lst[i].shape[0] - load_local(dts[i]).shape[0])


llm_obs = evaluate.load("llm-observatory/llm-observatory-eval", download_mode="force_redownload")

llm_obs.extract(
    model_name = ["llama3_8b_instruct"],
    model=model,
    task=llm_obs.task_specs[0],
    prob=False,
)

llm_obs.compute(
    models = ["llama3_8b_instruct"],
    tasks=llm_obs.task_specs,
    prob=False,
)

# from huggingface_hub import cached_assets_path
# from pathlib import Path

# # Get the cache path
# p = cached_assets_path("llm-observatory", namespace="test", subfolder="sandbox")
# p.mkdir(parents=True, exist_ok=True)

# # Write a test file
# f = p / "test.txt"
# f.write_text("Hello from llm-observatory cache!")

# print(f"✅ Wrote to {f}")
