import pandas as pd
import sys, os
import numpy as np
import json
from tqdm import tqdm
from functools import lru_cache
import gc

sys.path.append(os.path.join(os.getcwd(), "workspace"))
from workspace.common import *
import os
from concurrent.futures import ThreadPoolExecutor as Pool, as_completed
from concurrent.futures import (
    ThreadPoolExecutor,         # <- swap to threads (no pickling headaches)
    as_completed,               # iterate as results arrive
    wait, FIRST_EXCEPTION       # optional: bail out early on first error
)

def get_ground_truth(data, task_spec):
    return data[task_spec["variables"][0]].tolist()

@lru_cache(maxsize=None)
def load_dataset_shared(path: str):
    """
    Read the Parquet file once and hand out the same DataFrame
    to all worker threads.  All columns are kept.
    """
    return pd.read_parquet(path)          # <- all columns, per your requirement
# ----------------------------------------------------------------------------

def load_dataset(dataset):
    return pd.read_parquet(f"data/clean/{dataset}.parquet")

def task_extract(model_name, model, task_spec, check_cache=False, prob=False):
    """
    Run model evaluation on a benchmark task, supporting both marginal and conditional queries. 
    Save results to disk into a JSON file, containing both true values (from a ground truth dataset)
    and model's values (obtained from either Monte Carlo sampling, or from probabilities of next token).

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
    Saves a JSON file under `data/benchmark/` with the model predictions and true values.
    """
    
    dataset_name = task_spec['dataset'].split('/')[-1].split('.')[0]
    file_name = task_to_filename(model_name, task_spec)
    if check_cache and os.path.exists(os.path.join("data", "benchmark", file_name)):
        return None
    
    # Step 1: determine if query is marginal or conditional
    if "v_cond" in task_spec:
        ttyp = "hd"
    elif len(task_spec["variables"]) == 1:
        ttyp = "marginal"
    elif len(task_spec["variables"]) == 2:
        ttyp = "conditional"
    
    data = load_dataset_shared(task_spec["dataset"])
    
    # If model == None, we draw synthetic "model" values from a reference dataset
    if model is None:
        mc_data = pd.read_parquet(model_name)
        n_mc = 128

    if ttyp in ["hd_old", "hd"]:
        if prob:
            levels = gen_prob_lvls()
        else:
            levels = sorted(data[task_spec["v_out"]].unique().tolist())
    else:
        if prob:
            levels = gen_prob_lvls()
            q_levels = data[task_spec["variables"][0]].unique().tolist()
            q_levels = [x for x in q_levels if x is not None]
        else:
            levels = data[task_spec["variables"][0]].unique().tolist()

    results = []
    if ttyp == "marginal":
        # ground‑truth values
        true_vals = get_ground_truth(data, task_spec)

        # model values
        model_vals, model_texts = extract_pv(
            task_spec["prompt"],
            levels,
            model_name,
            model,
            task_spec
        )

        # compress to save memory (weights argument is now always None)
        true_vals, _ = compress_vals(true_vals, None)

        results.append({
            "condition": "All",
            "true_vals": true_vals,
            "model_vals": model_vals,
            "model_texts": model_texts
        })

    elif ttyp == "conditional":
        # conditional query – iterate over levels of the conditioning variable
        cond_range = data[task_spec["variables"][1]].unique()
        if "cond_range" in task_spec:
            lo, hi = task_spec["cond_range"]
            cond_range = cond_range[(cond_range >= lo) & (cond_range <= hi)]
        
        if "subset" in task_spec:
            v_sub = task_spec["subset"][0]
            sub_type = task_spec["subset"][1]
            target_val = task_spec["subset"][2]
            if sub_type == "lwr":
                data = data[data[v_sub] >= target_val]
            elif sub_type == "levels":
                data = data[data[v_sub].isin(target_val)]
            
            # update levels after subsetting
            if prob:
                q_levels = data[task_spec["variables"][0]].unique().tolist()
                q_levels = [x for x in q_levels if x is not None]
            else: 
                levels = data[task_spec["variables"][0]].unique().tolist()
        
        for cond in tqdm(cond_range):
            filtered_data = data[data[task_spec["variables"][1]] == cond]
            
            true_vals = get_ground_truth(filtered_data, task_spec)
            
            if prob:
                prompt = [task_spec["prompt_prob"].format(cond, qlvl) for qlvl in q_levels]
                pos_ans = "(possible levels of " + task_spec["variables"][0] +  " are: " + ", ".join(q_levels) + ")"
            else:
                prompt = task_spec["prompt"].format(cond)
                pos_ans = None

            # model values
            model_vals, model_weights, model_texts = extract_pv(
                prompt,
                levels,
                model_name,
                model,
                task_spec,
                pos_ans=pos_ans
            )

            if prob:
                model_weights = decode_prob_matrix(levels, model_weights)
                model_vals = q_levels

            if "weight" in filtered_data.columns:
                weights = filtered_data["weight"].tolist()
            else:
                weights = [1] * len(true_vals)
            
            if not true_vals or not weights:
                breakpoint()
            true_vals, weights = compress_vals(true_vals, weights)

            results.append({
                "condition": cond.tolist() if hasattr(cond, "tolist") else cond,
                "true_vals": true_vals,
                "true_weights": weights,
                "n_data": len(filtered_data),
                "total_weight": sum(weights),
                "model_vals": model_vals,
                "model_weights": model_weights,
                "model_texts": model_texts
            })
    elif ttyp == "hd_old":

        cond_vars = task_spec["v_cond"]
        out_var = task_spec["v_out"]
        
        # fit lightgbm model to get conditional probabilities (ground truth)
        preds = fit_lgbm(data, out_var, cond_vars, wgh_col="weight")
        data["lgbm_pred"] = preds
        
        # get all combinations
        cond_df = data[cond_vars].drop_duplicates().reset_index(drop=True)
        cond_df["llm_pred"] = np.nan
        for i, cond_row in tqdm(cond_df.iterrows(), total=len(cond_df)):
            
            # get the natural language prompt for the row
            row_prompt = promptify(out_var, cond_vars, cond_row, dataset_name, prob=prob)
            # model values
            model_vals, model_weights, model_texts = extract_pv(row_prompt, levels, model_name, model, task_spec)
            cond_df.at[i, "llm_pred"] = model_weights[1] # get the P(Y = 1 | X = x)

        data = data.merge(cond_df, on=cond_vars, how="left")
    elif ttyp == "hd":
        cond_vars = task_spec["v_cond"]
        out_var = task_spec["v_out"]
        
        # fit lightgbm model to get conditional probabilities (ground truth)
        preds = fit_lgbm(data, out_var, cond_vars, wgh_col="weight")
        data["lgbm_pred"] = preds
        
        # get all combinations
        cond_df = data[cond_vars].drop_duplicates().reset_index(drop=True)
        cond_df["llm_pred"] = np.nan
        
        prompts = [
            promptify(out_var, cond_vars, row, dataset_name, prob=prob)  # type: ignore[name‑defined]
            for _, row in cond_df.iterrows()
        ]
        
        model_vals, model_weights, model_texts = extract_pv_batch(
            prompts=prompts,
            levels=levels,
            model_name=model_name,
            model=model,
            task_spec=task_spec,
            prob=prob,
        )
        if prob:
            llm_probs = [decode_prob_lvl(model_vals, probs) for probs in model_weights]
        else:
            llm_probs = [x[1] for x in model_weights]
        
        cond_df['llm_pred'] = llm_probs  # get the P(Y = 1 | X = x)
        data = data.merge(cond_df, on=cond_vars, how="left")

    # save to disk
    os.makedirs(os.path.join("data", "benchmark"), exist_ok=True)
    
    if prob:
        file_name = "PROB_" + file_name
    if ttyp == "hd":
        # save the full dataset with predictions
        data.to_parquet(os.path.join("data", "benchmark-hd", file_name))
    else:
        with open(os.path.join("data", "benchmark", file_name), "w") as f:
            json.dump(results, f, indent=4)


# One‑liner wrapper so the pool receives just the spec
def _run_task(task_spec):
    # model_name and model must exist inside the worker.
    # If they're picklable you can pass them in via globals,
    # otherwise see the follow‑up notes below.
    return task_extract(model_name, model, task_spec, check_cache=True, prob=True)

model_name = "gpt-4.1"
model = load_model(model_name)

n_workers = min(os.cpu_count(), len(task_specs_hd))
n_workers = 6 #n_workers // 2

def run_tasks(task_specs, *,
              max_workers: int | None = None,
              stop_on_first_error: bool = False):
    """
    Execute all task_specs concurrently and surface exceptions.

    Returns a list with one result per spec (in the same order).
    Raises the *first* exception unless stop_on_first_error=False,
    in which case it returns (results, exceptions).
    """
    if not task_specs:           # nothing to do
        return []

    # Deduce a sensible worker count
    n_cpu = os.cpu_count() or 1
    max_workers = max_workers or min(len(task_specs), n_cpu)
    assert max_workers > 0, "max_workers must be ≥1"

    results      = [None] * len(task_specs)
    exceptions   = [None] * len(task_specs)
    print(f"Running {len(task_specs)} tasks with {max_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_run_task, spec): idx
            for idx, spec in enumerate(task_specs)
        }

        # as_completed lets us surface errors early
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                exceptions[idx] = exc
                if stop_on_first_error:
                    # Cancel the remaining work and reraise
                    for f in future_to_idx:
                        f.cancel()
                    raise

    if any(exceptions):
        # Bubble up the *first* error so caller sees traceback,
        # but still return the others if you need them.
        first_err = next(e for e in exceptions if e is not None)
        raise first_err

    return results

# task subset
task_subset = []
# indices = [ 0,  1,  2,  3,  4,  6,  7, 13, 14, 15, 16, 17, 18, 19, 20, 26, 27, 28, 29, 31, 32, 33, 39, 40, 42,
#   44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 68, 69, 70, 72, 73, 74, 75, 76,
#   77, 78, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91]
# web indices
# indices = [39, 42, 56, 61, 62, 63, 64, 65, 66, 69, 70]

# missing_indices = list(set(range(len(task_specs_hd))) - set(indices))

indices = list(range(len(task_specs)))

for index in indices:
    task_subset.append(task_specs[index])

run_tasks(task_subset, max_workers=n_workers, stop_on_first_error=False)
