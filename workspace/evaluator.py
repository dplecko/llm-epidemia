import pandas as pd
import sys, os
import numpy as np
import json
import pdb
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "workspace"))
from model_load import load_model
from evaluator_helpers import extract_pv, compress_vals, xgb_conditional_prob  # removed d2d_wgh_col
from task_spec import task_specs

def get_ground_truth(data, task_spec):
    return data[task_spec["variables"][0]].tolist()

def load_dataset(dataset):
    return pd.read_parquet(f"data/clean/{dataset}.parquet")
        

def evaluator(model_name, model, task_spec, check_cache=False):
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
    file_name = f"{model_name.split('/')[-1].split('.')[0]}_{task_spec['dataset'].split('/')[-1].split('.')[0]}_{task_spec['variables'][0]}"
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

    levels = data[task_spec["variables"][0]].unique().tolist()

    results = []
    if marginal:
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

    elif len(task_spec["variables"]) == 2:
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
        
        for cond in tqdm(cond_range):
            filtered_data = data[data[task_spec["variables"][1]] == cond]
            
            true_vals = get_ground_truth(filtered_data, task_spec)
            
            # model values
            model_vals, model_weights, model_texts = extract_pv(
                task_spec["prompt"].format(cond),
                levels,
                model_name,
                model,
                task_spec
            )

            if "weight" in filtered_data.columns:
                weights = filtered_data["weight"].tolist()
            else:
                weights = [1] * len(true_vals)
            
            if not true_vals or not weights:
                pdb.set_trace()
            true_vals, weights = compress_vals(true_vals, weights)

            results.append({
                "condition": cond.tolist() if hasattr(cond, "tolist") else cond,
                "true_vals": true_vals,
                "true_weights": weights,
                "n_data": len(filtered_data),
                "model_vals": model_vals,
                "model_weights": model_weights,
                "model_texts": model_texts
            })
    else:
        # TODO create a single promopt:
        # task_spec["prompt"] = generate_promt()  # use the nsduh_con and nsduh_out
        # multi variable conditioning
        # assume varibles are ordered according to the generated prompt
        cond_vars = task_spec["variables"][1:] 
        # get all combinations
        cond_range = (
            data[cond_vars]                # keep only the conditioning columns
            .drop_duplicates()             # keep one row per unique combo
            .itertuples(index=False, name=None)  # → iterator of plain tuples
        )
        cond_range = list(cond_range)      # materialise as list of tuples
        # e.g. [(x1, y1), (x1, y2), (x2, y1), ...]
        
        for cond in tqdm(cond_range):
            # Build a boolean mask for rows matching *all* fields in cond
            mask = True
            for col, val in zip(cond_vars, cond):
                mask &= (data[col] == val)
            filtered_data = data[mask]
            
            # code repetition, move it to a separate function?
            
            # do xgboost here
            # TODO test this
            true_vals = xgb_conditional_prob(target=task_spec["variables"][0], df=filtered_data,)
            
            # model values
            model_vals, model_weights, model_texts = extract_pv(
                task_spec["prompt"].format(cond),
                levels,
                model_name,
                model,
                task_spec
            )

            if "weight" in filtered_data.columns:
                weights = filtered_data["weight"].tolist()
            else:
                weights = [1] * len(true_vals)
            
            if not true_vals or not weights:
                pdb.set_trace()
            true_vals, weights = compress_vals(true_vals, weights)

            results.append({
                "condition": cond.tolist() if hasattr(cond, "tolist") else cond,
                "true_vals": true_vals,
                "true_weights": weights,
                "n_data": len(filtered_data),
                "model_vals": model_vals,
                "model_weights": model_weights,
                "model_texts": model_texts
            })

    # save to disk
    os.makedirs(os.path.join("data", "benchmark"), exist_ok=True)
    with open(os.path.join("data", "benchmark", file_name), "w") as f:
        json.dump(results, f, indent=4)

# if __name__ == "__main__":
#     d2d = False

#     if d2d:
#         task_sel = range(2)
#         models = ["data/clean/nhanes.parquet", "data/clean/gss.parquet"]
#     else:
#         task_sel = [0] # range(1) # range(len(task_specs))
#         models = ["llama3_8b_instruct"]

#     for model_name in models:
#         model = load_model(model_name)
#         for i in task_sel:
#             evaluator(model_name, model, task_specs[i], check_cache=True)

model_name = "llama3_8b_instruct"
model = load_model(model_name)
for i in np.arange(0, 12):
    evaluator(model_name, model, task_specs[i], check_cache=False)

# meps: 37, 47
# nhanes: 46, 50
# nsduh: ..., ...
# scf: ..., ...
# df.loc[df["employment_status"].isna(), "age"].value_counts()
# df.loc[df["employer"].isna(), "employment_status"].value_counts()

from helpers import model_name, model_unname
from build_eval_df import build_eval_df

eval_df = build_eval_df(["llama3_8b_instruct"], task_specs[0:1])