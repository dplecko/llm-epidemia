
import pandas as pd
import numpy as np
from itertools import combinations
import random

model_name_map = {
    "llama3_8b_instruct": "LLama3 8B",
    "llama3_70b_instruct": "LLama3 70B",
    "mistral_7b_instruct": "Mistral 7B",
    "deepseek_7b_chat": "DeepSeek 7B",
    "phi4": "Phi4",
    "gemma3_27b_instruct": "Gemma3 27B",
    "nhanes": "NHANES",
    "gss": "GSS"
}

model_display_map = {v: k for k, v in model_name_map.items()}

def hd_taskgen(out_spec, cond_spec, d_min=2, d_max=5, max_per_dim=100):
    random.seed(42)
    tasks_hd = []
    for v_out in out_spec.keys():
        for r in range(d_min, min(d_max, len(cond_spec)) + 1):
            cnt = 0
            all_combinations = list(combinations(cond_spec.keys(), r))
            random.shuffle(all_combinations)
            for v_cond in all_combinations:
                if cnt > max_per_dim:
                    break
                else:
                    cnt += 1
                tasks_hd.append({
                    "v_out": v_out,
                    "v_cond": list(v_cond)
                })
    return tasks_hd

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

def weighted_corr(x, y, w):
    
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0
    
    # Weighted means
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    
    # Weighted covariance
    cov = np.sum(w * (x - mx) * (y - my))
    
    # Weighted variances
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    
    # Weighted correlation
    return cov / np.sqrt(vx * vy)

def weighted_L1(x, y, w):
    return np.sum(np.abs(x - y) * w) / np.sum(w)

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
