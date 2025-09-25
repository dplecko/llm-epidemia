
import pandas as pd
import subprocess
import re

model_name_map = {
    "llama3_8b_instruct": "LLama3 8B",
    "llama3_8b_instruct_sft": "LLama3 8B SFT",
    "llama3_8b": "LLama3 8B",
    "llama3_70b_instruct": "LLama3 70B",
    "llama3_70b": "LLama3 70B",
    "mistral_7b_instruct": "Mistral 7B",
    "mistral_7b_instruct_sft": "Mistral 7B SFT",
    "mistral_7b": "Mistral 7B",
    "deepseek_7b_chat": "DeepSeek 7B",
    "deepseek_7b": "DeepSeek 7B",
    "phi4": "Phi4",
    "gemma3_27b_instruct": "Gemma3 27B",
    "gemma3_27b": "Gemma3 27B",
    "deepseekR1_32b": "DeepSeek R1 (Qwen 32B)",
    "gpt-4.1": "GPT 4.1",
    "o4-mini": "o4 mini"
}

model_colors = {
    "LLama3 8B": "#1f77b4",
    "LLama3 70B": "#ff7f0e",
    "Mistral 7B": "#2ca02c",
    "DeepSeek 7B": "#17becf",
    "Phi4": "#d62728",
    "Gemma3 27B": "#9467bd",
    "DeepSeek R1 (Qwen 32B)": "#8c564b",
    "GPT 4.1": "#e377c2",
    "o4 mini": "#7f7f7f"
}

dts_map = {
    "nhanes": "NHANES",
    "gss": "GSS",
    "brfss": "BRFSS",
    "nsduh": "NSDUH",
    "acs": "ACS",
    "edu": "IPEDS",
    "fbi_arrests": "FBI Arrests",
    "labor": "BLS",
    "meps": "MEPS",
    "scf": "SCF",
}

var_map = {
    # Outcomes
    "diabetes": "Diabetes",
    "high_bp": "High Blood Pressure",
    "depression": "Depression",
    "insured": "Health Insurance",
    "cig_monthly": "Cigarette Use (Last 30d)",
    "mj_ever": "Marijuana Use",
    "coc_ever": "Cocaine Use",
    # Conditions
    "house_own": "Home Ownership",
    "age_group": "Age",
    "age": "Age",
    "education": "Education",
    "education_years": "Education",
    "edu": "Education",
    "race": "Race",
    "sex": "Sex",
    "income": "Income"
}

model_display_map = {v: k for k, v in model_name_map.items()}

def task_to_filename(model_name, task_spec):
    dataset_name = task_spec['dataset'].split('/')[-1].split('.')[0]
    if "v_cond" in task_spec:
        cond_vars_str = "_".join(task_spec["v_cond"])
        file_name = f"{model_name}_{dataset_name}_{task_spec['v_out']}_{cond_vars_str}.parquet"
    else:
        file_name = f"{model_name}_{dataset_name}_{task_spec['variables'][0]}"
        if len(task_spec["variables"]) > 1:
            file_name += f"_{task_spec['variables'][1]}"
        file_name = file_name + ".json"
    return file_name

def hd_taskname(task):
    dataset = task['dataset'].split('/')[-1].split('.')[0]
    dataset = dts_map.get(dataset, dataset)
    out = var_map.get(task['v_out'], task['v_out'])
    cond = pd.Series(task['v_cond']).map(var_map).tolist()
    # pdbpp.set_trace()
    cond = ", ".join(cond)
    return dataset + ": " + out + " by " + cond

def model_name(mod):
    if isinstance(mod, pd.Series):
        return mod.map(lambda x: model_name_map.get(x, x))
    elif isinstance(mod, list):
        return [model_name_map.get(m, m) for m in mod]
    return model_name_map.get(mod, mod)

def model_unname(mod):
    if isinstance(mod, pd.Series):
        return mod.map(lambda x: model_display_map.get(x, x))
    elif isinstance(mod, list):
        return [model_display_map.get(m, m) for m in mod]
    return model_display_map.get(mod, mod)

def dat_name_clean(path):
    base = path.split("/")[-1]
    return re.sub(r"\.parquet$", "", base)

def bin_labels(breaks, unit="$", exact=False, last_plus = False):
    if exact:
        if last_plus:
            labels = [f"{b} {unit}" for b in breaks[:-1]]
            labels.append(f"{breaks[-1]}+ {unit}")
        else:
            labels = [f"{b} {unit}" for b in breaks]
    else:
        labels = [f"< {breaks[0]} {unit}"]
        for i in range(1, len(breaks)):
            labels.append(f"{breaks[i-1]}–{breaks[i]} {unit}")
        labels.append(f"{breaks[-1]}+ {unit}")
    return labels

def load_dts(task, cache_dir=None):
    if cache_dir is not None:
        # Hosted mode → load from HF dataset
        from datasets import load_dataset
        dataset_id = "llm-observatory/llm-observatory"
        config = dat_name_clean(task["dataset"])
        data_files = f"hf://datasets/llm-observatory/llm-observatory/data/{config}.parquet"
        dts = load_dataset("parquet", data_files=data_files, split="train")
        return dts.to_pandas()
    else:
        # local mode: load parquet files from data/clean
        return pd.read_parquet(task["dataset"])

def sync_bench():
    cmd = [
        "rsync", "-avz", "--update", "--progress",
        "-e", "ssh",
        "eb0:~/llm-epidemia/data/benchmark/",
        "~/trust/llm-epidemia/data/benchmark/"
    ]
    subprocess.run(" ".join(cmd), shell=True)
