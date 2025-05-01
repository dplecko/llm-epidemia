import pandas as pd
import sys, os
import numpy as np
import json
import pdb
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "py"))
from model_load import load_model
from py.evaluator_helpers import extract_pv, compress_vals  # removed d2d_wgh_col
from task_spec import task_specs


def get_ground_truth(data, task_spec):
    if task_spec["levels"] is not None and not pd.api.types.is_numeric_dtype(data[task_spec["variables"][0]]):
        level_map = {v: i for i, group in enumerate(task_spec["levels"]) for v in group}
        true_vals = data[task_spec["variables"][0]].map(level_map).tolist()
    else:
        true_vals = data[task_spec["variables"][0]].tolist()
    return true_vals
        

def evaluator(model_name, model, tokenizer, task_spec, check_cache=False):
    """
    Run model evaluation on a benchmark task, supporting both marginal and conditional queries. 
    Save results to disk into a JSON file, containing both true values (from a ground truth dataset)
    and model's values (obtained from either Monte Carlo sampling, or from probabilities of next token).

    Note
    ----
    Weight columns (`wgh_col`) have been completely removed for simplicity; every sample
    now carries equal importance.

    Args
    ----
    model_name (str): Name of the model, used in output file naming and batch size config.
    model: Language model to evaluate.
    tokenizer: Tokenizer for input preparation and decoding.
    task_spec (dict): Task specification containing:
        - "prompt" (str): Base prompt or prompt template.
        - "variables" (List[str]): Outcome variable, optionally with a conditioning variable.
        - "dataset" (str): Path to the dataset parquet file.
        - "levels" (List[List[str]] | None): Discrete level groupings for classification.
        - "second_prompt" (str, optional): Follow‑up question (for "story" mode).
    check_cache (bool): If *True* and the task was already solved for the model, skip the run.

    Outputs
    -------
    Saves a JSON file under `data/results/benchmark/` with the model predictions and true values.
    """
    file_name = f"{model_name.split('/')[-1].split('.')[0]}_{task_spec['mode']}_{task_spec['dataset'].split('/')[-1].split('.')[0]}_{task_spec['variables'][0]}"
    if len(task_spec["variables"]) > 1:
        file_name += f"_{task_spec['variables'][1]}"
    file_name = file_name + ".json"
    
    if check_cache and os.path.exists(os.path.join("data", "benchmark", file_name)):
        return None
    
    # Step 1: determine if query is marginal or conditional
    marginal = len(task_spec["variables"]) == 1
    
    data = pd.read_parquet(task_spec["dataset"])
    
    # If model == None, we draw synthetic "model" values from a reference dataset
    if model is None:
        mc_data = pd.read_parquet(model_name)
        n_mc = 128

    results = []
    if marginal:
        # ground‑truth values
        true_vals = get_ground_truth(data, task_spec)

        # model values
        if model is not None:
            model_vals, model_texts = extract_pv(
                task_spec["prompt"],
                task_spec["levels"],
                model_name,
                model,
            )
        else:
            # TODO: do we need this?
            # make sure synthetic sample respects the observed range
            t_max = min(max(true_vals), mc_data[task_spec["variables"][0]].values.max())
            t_min = max(min(true_vals), mc_data[task_spec["variables"][0]].values.min())
            mc_data = mc_data.query(f"{task_spec['variables'][0]} >= @t_min & {task_spec['variables'][0]} <= @t_max - 1")
            model_vals = np.random.choice(
                mc_data[task_spec["variables"][0]].values,
                size=n_mc,
                replace=True
            ).tolist()
            model_texts = None

        # compress to save memory (weights argument is now always None)
        true_vals, _ = compress_vals(true_vals, None)

        results.append({
            "condition": "All",
            "true_vals": true_vals,
            "model_vals": model_vals,
            "model_texts": model_texts
        })

    else:
        # conditional query – iterate over levels of the conditioning variable
        cond_range = data[task_spec["variables"][1]].unique()
        if "cond_range" in task_spec:
            lo, hi = task_spec["cond_range"]
            cond_range = cond_range[(cond_range >= lo) & (cond_range <= hi)]
        
        for cond in tqdm(cond_range):
            filtered_data = data[data[task_spec["variables"][1]] == cond]
            
            true_vals = get_ground_truth(filtered_data, task_spec)
            
            # model values
            if model is None:
                # TODO: do we need this?
                filter_mc = mc_data[mc_data[task_spec["variables"][1]] == cond]
                model_vals = np.random.choice(
                    filter_mc[task_spec["variables"][0]].values,
                    size=n_mc,
                    replace=True
                ).tolist()
                if task_spec["levels"] is not None and not pd.api.types.is_numeric_dtype(data[task_spec["variables"][0]]):
                    level_map = {v: i for i, group in enumerate(task_spec["levels"]) for v in group}
                    model_vals = [level_map.get(v, None) for v in model_vals]
                model_texts = None
            else:
                model_vals, model_texts = extract_pv(
                    task_spec["prompt"].format(cond),
                    task_spec["levels"],
                    model_name,
                    model,
                )

            true_vals, _ = compress_vals(true_vals, None)

            results.append({
                "condition": cond.tolist() if hasattr(cond, "tolist") else cond,
                "true_vals": true_vals,
                "model_vals": model_vals,
                "model_texts": model_texts
            })

    # save to disk
    os.makedirs(os.path.join("data", "benchmark"), exist_ok=True)
    with open(os.path.join("data", "benchmark", file_name), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    d2d = False

    if d2d:
        task_sel = range(2)
        models = ["data/clean/nhanes.parquet", "data/clean/gss.parquet"]
    else:
        task_sel = range(len(task_specs))
        models = ["llama3_70b_instruct"]

    for model_name in models:
        tokenizer, model, is_instruct = load_model(model_name)
        for i in task_sel:
            evaluator(model_name, model, tokenizer, task_specs[i], check_cache=True)
