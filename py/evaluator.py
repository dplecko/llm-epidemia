
import pandas as pd
import sys, os
import numpy as np
import json
import pdb
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "py"))
from model_load import load_model
from evaluator_helpers import extract_pv, d2d_wgh_col, compress_vals
from task_spec import task_specs


def evaluator(model_name, model, task_spec, check_cache=False):
    """
    Run model evaluation on a benchmark task, supporting both marginal and conditional queries. 
    Save results to disk into a JSON file, containing both true values (from a ground truth dataset)
    and model's values (obtained from either Monte Carlo sampling, or from probabilities of next token).

    Args:
        model_name (str): Name of the model, used in output file naming and batch size config.
        model: Language model to evaluate.
        task_spec (dict): Task specification containing:
            - "prompt" (str): Base prompt or prompt template.
            - "variables" (List[str]): Outcome variable, optionally with a conditioning variable.
            - "dataset" (str): Path to the dataset CSV.
            - "levels" (List[List[str]] or None): Discrete level groupings for classification.
            - "mode" (str): One of {"logits", "sample", "story"}.
            - "second_prompt" (str, optional): Follow-up question (for "story" mode).
            - "wgh_col" (str or None): Optional column name for sample weights.
        check_cache (logical): Check if the task was already solved for the model; skip the run if yes.

    Outputs:
        Saves a JSON file under `data/results/benchmark/` with the model predictions, true values, and optional weights.
    """
    file_name = f"{model_name.split('/')[-1].split('.')[0]}_{task_spec['mode']}_{task_spec['dataset'].split('/')[-1].split('.')[0]}_{task_spec['variables'][0]}"
    if len(task_spec["variables"]) > 1:
        file_name += f"_{task_spec['variables'][1]}"
    file_name = file_name + ".json"
    
    if check_cache and os.path.exists(os.path.join("data", "benchmark", file_name)):
        return None
    
    # Step 1: check if query is marginal or conditional
    if len(task_spec["variables"]) == 1:
        marginal = True
    else:
        marginal = False
    
    data = pd.read_parquet(task_spec["dataset"])
    if model is None:
        mc_data = pd.read_parquet(model_name)
        n_mc = 128
        mc_wgh_col = d2d_wgh_col(model_name)

    results = []
    if marginal:
        if task_spec["levels"] is not None and not pd.api.types.is_numeric_dtype(data[task_spec["variables"][0]]):
            level_map = {v: i for i, group in enumerate(task_spec["levels"]) for v in group}
            true_vals = data[task_spec["variables"][0]].map(level_map).tolist()
        else:
            true_vals = data[task_spec["variables"][0]].tolist()

        if task_spec["wgh_col"] is not None:
            wghs = data[task_spec["wgh_col"]].tolist()
        else:
            wghs = None

        if model is not None:
            model_vals, model_texts = extract_pv(
                task_spec["prompt"], task_spec["levels"], task_spec["mode"], 
                model_name, model, second_prompt=task_spec.get("second_prompt", None)
            )
        else:

            # do the trimming for top-coded values
            t_max = min(max(true_vals), mc_data[task_spec["variables"][0]].values.max())
            t_min = max(min(true_vals), mc_data[task_spec["variables"][0]].values.min())

            # trim model values
            mc_data = mc_data[mc_data[task_spec["variables"][0]] <= t_max - 1]
            mc_data = mc_data[mc_data[task_spec["variables"][0]] >= t_min]

            # trim true values
            mask = [(t_min <= i <= t_max - 1) for i in true_vals]
            if wghs is not None:
                wghs = [i for i, keep in zip(wghs, mask) if keep]
            true_vals = [i for i, keep in zip(true_vals, mask) if keep]
            
            if mc_wgh_col in mc_data.columns:
                mc_wghs = mc_data[mc_wgh_col].values
                mc_wghs = mc_wghs / mc_wghs.sum()
            else:
                mc_wghs = None

            model_vals = np.random.choice(mc_data[task_spec["variables"][0]].values, size=n_mc, replace=True, p=mc_wghs).tolist()
            model_texts = None

        # compress the true values to save memory
        true_vals, wghs = compress_vals(true_vals, wghs)

        results.append({
            "condition": "All",
            "true_vals": true_vals,
            "weights": wghs,
            "model_vals": model_vals,
            "model_texts": model_texts,
        })

    else:
        # iterate over different levels of the conditioning variable
        cond_range = data[task_spec["variables"][1]].unique()
        if "cond_range" in task_spec:
            lo, hi = task_spec["cond_range"]
            cond_range = cond_range[(cond_range >= lo) & (cond_range <= hi)]
        
        for cond in tqdm(cond_range):
            # filter the dataset for the current level
            filtered_data = data[data[task_spec["variables"][1]] == cond]
            
            if task_spec["levels"] is not None and not pd.api.types.is_numeric_dtype(data[task_spec["variables"][0]]):
                level_map = {v: i for i, group in enumerate(task_spec["levels"]) for v in group}
                true_vals = filtered_data[task_spec["variables"][0]].map(level_map).tolist()
            else:
                true_vals = filtered_data[task_spec["variables"][0]].tolist()
            
            if task_spec["wgh_col"] is not None:
                wghs = filtered_data[task_spec["wgh_col"]].tolist()
            else:
                wghs = None

            if model is None:
                filter_mc = mc_data[mc_data[task_spec["variables"][1]] == cond]
                if mc_wgh_col in filter_mc.columns:
                    mc_wghs = filter_mc[mc_wgh_col].values
                    mc_wghs = mc_wghs / mc_wghs.sum()
                else:
                    mc_wghs = None

                model_vals = np.random.choice(filter_mc[task_spec["variables"][0]].values, size=n_mc, replace=True, p=mc_wghs).tolist()
                if task_spec["levels"] is not None and not pd.api.types.is_numeric_dtype(data[task_spec["variables"][0]]):
                    model_vals = [level_map.get(v, None) for v in model_vals]
                model_texts = None
            else:
                model_vals, model_texts = extract_pv(
                    task_spec["prompt"].format(cond), task_spec["levels"], task_spec["mode"], 
                    model_name, model, second_prompt=task_spec.get("second_prompt", None)
                )

            true_vals, wghs = compress_vals(true_vals, wghs)
            results.append({
                "condition": cond.tolist() if hasattr(cond, "tolist") else cond,
                "true_vals": true_vals,
                "weights": wghs,
                "model_vals": model_vals,
                "model_texts": model_texts,
            })

    with open(os.path.join("data", "benchmark", file_name), "w") as f:
        json.dump(results, f, indent=4)

d2d = False

if d2d:
    task_sel = range(2)
    models = ["data/clean/nhanes.parquet", "data/clean/gss.parquet"]
else:
    task_sel = range(len(task_specs))
    models = ["gpt2"]
    # models = ["llama3_8b_instruct", "mistral_7b_instruct"]
    # models = ["llama3_8b_instruct", "mistral_7b_instruct", "phi4", "gemma3_27b_instruct", "llama3_70b_instruct"]


for model_name in models:
    model = load_model(model_name)
    for i in task_sel:
        evaluator(model_name, model, task_specs[i], check_cache=True)
    