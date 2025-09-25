
import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import models

# load LLM Observatory infrastructure
llm_obs = evaluate.load("llm-observatory/llm-observatory-eval")

# prepare the model
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
hf_model = models.HuggingFaceModel(model, tokenizer)

# extract model answers
llm_obs.extract(model_name = "llama3_8b_instruct", model=hf_model, task=llm_obs.task_specs[0])
llm_obs.compute(models = ["llama3_8b_instruct"], tasks=llm_obs.task_specs[0:1])

# HF dataset loading code
# ds = load_dataset("parquet",
#                   data_files="hf://datasets/llm-observatory/llm-observatory/data/meps.parquet",
#                   split="train")

# df = load_dataset("llm-observatory/llm-observatory", "meps", split="train", trust_remote_code=True).to_pandas()

# df_lst = []
# dts = ["brfss", "nhanes", "nsduh", "gss", "fbi_arrests", "edu", "labor", "scf", "acs"]
# for i in range(len(dts)):
#     df_lst.append(load_dataset("llm-observatory/llm-observatory", dts[i], split="train", trust_remote_code=True).to_pandas())

# def load_local(dataset):
#     return pd.read_parquet(f"data/clean/{dataset}.parquet")

# for i in range(len(dts)):
#     print(dts[i], " ", df_lst[i].shape[0] - load_local(dts[i]).shape[0])