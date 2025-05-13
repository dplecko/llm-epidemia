import pandas as pd
import sys, os
import numpy as np
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "workspace"))
from model_load import load_model
from evaluator_helpers import extract_pv, compress_vals, extract_pv_batch
from task_spec import task_specs, task_specs_hd
from helpers import task_to_filename
from hd_helpers import fit_lgbm, promptify

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
    
    data = pd.read_parquet(task_spec["dataset"])
    
    # If model == None, we draw synthetic "model" values from a reference dataset
    if model is None:
        mc_data = pd.read_parquet(model_name)
        n_mc = 128

    if ttyp == "hd":
        levels = sorted(data[task_spec["v_out"]].unique().tolist())
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
            levels = data[task_spec["variables"][0]].unique().tolist()
        
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
        # TODO create a single promopt:
        # task_spec["prompt"] = generate_promt()  # use the nsduh_con and nsduh_out
        # multi variable conditioning
        # assume varibles are ordered according to the generated prompt

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
            row_prompt = promptify(out_var, cond_vars, cond_row, dataset_name)
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
            promptify(out_var, cond_vars, row, dataset_name)  # type: ignore[name‑defined]
            for _, row in cond_df.iterrows()
        ]
        
        model_vals, model_weights, model_texts = extract_pv_batch(
            prompts=prompts,
            levels=levels,
            model_name=model_name,
            model=model,
            task_spec=task_spec,
        )
        llm_probabilities = [x[1] for x in model_weights]
        cond_df['llm_pred'] = llm_probabilities  # get the P(Y = 1 | X = x)
        data = data.merge(cond_df, on=cond_vars, how="left")

    # save to disk
    os.makedirs(os.path.join("data", "benchmark"), exist_ok=True)
    
    if ttyp == "hd":
        # save the full dataset with predictions
        data.to_parquet(os.path.join("data", "benchmark", file_name))
    else:
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

# models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", "gemma3_27b_instruct"]
# for model_name in models:
#     print("\nEntering model: ", model_name, "\n")
#     model = load_model(model_name)
#     for i in range(len(task_specs)):
#         evaluator(model_name, model, task_specs[i], check_cache=False)

model_name = "gemma3_27b_instruct"
model = load_model(model_name)
for i in range(len(task_specs_hd)):
    evaluator(model_name, model, task_specs_hd[i], check_cache=False)

