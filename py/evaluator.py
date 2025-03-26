
import pandas as pd
import sys, os
import json
import pdb
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "py"))
from model_load import load_model
from py.evaluator_helpers import extract_pv
from task_spec import task_specs

def evaluator(model_name, model, tokenizer, task_spec):
    """
    Run model evaluation on a benchmark task, supporting both marginal and conditional queries. 
    Save results to disk into a JSON file, containing both true values (from a ground truth dataset)
    and model's values (obtained from either Monte Carlo sampling, or from probabilities of next token).

    Args:
        model_name (str): Name of the model, used in output file naming and batch size config.
        model: Language model to evaluate.
        tokenizer: Tokenizer for input preparation and decoding.
        task_spec (dict): Task specification containing:
            - "prompt" (str): Base prompt or prompt template.
            - "variables" (List[str]): Outcome variable, optionally with a conditioning variable.
            - "dataset" (str): Path to the dataset CSV.
            - "levels" (List[List[str]] or None): Discrete level groupings for classification.
            - "mode" (str): One of {"logits", "sample", "story"}.
            - "second_prompt" (str, optional): Follow-up question (for "story" mode).
            - "wgh_col" (str or None): Optional column name for sample weights.

    Outputs:
        Saves a JSON file under `data/results/benchmark/` with the model predictions, true values, and optional weights.
    """
    # Step 1: check if query is marginal or conditional
    if len(task_spec["variables"]) == 1:
        marginal = True
    else:
        marginal = False
    
    # Step 2: check if the outcome is binary
    data = pd.read_parquet(task_spec["dataset"])
     
    results = []
    if marginal:
        if task_spec["levels"] is not None:
            true_vals = data[task_spec["variables"][0]].isin(task_spec["levels"][0])
        else:
            true_vals = data[task_spec["variables"][0]]

        model_vals = extract_pv(
            task_spec["prompt"], task_spec["levels"], task_spec["mode"], 
            model_name, model, tokenizer, second_prompt=task_spec.get("second_prompt", None)
        )

        results.append({
            "true_vals": true_vals,
            "model_vals": model_vals,
        })

    else:
        # iterate over different levels of the conditioning variable
        for cond in tqdm(data[task_spec["variables"][1]].unique()):
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

            model_vals, model_texts = extract_pv(
                task_spec["prompt"].format(cond), task_spec["levels"], task_spec["mode"], 
                model_name, model, tokenizer, second_prompt=task_spec.get("second_prompt", None)
            )

            results.append({
                "condition": cond.tolist() if hasattr(cond, "tolist") else cond,
                "true_vals": true_vals,
                "weights": wghs,
                "model_vals": model_vals,
                "model_texts": model_texts,
            })

    file_name = f"{model_name}_{task_spec['mode']}_{task_spec['dataset'].split('/')[-1].split('.')[0]}_{task_spec['variables'][0]}"
    if len(task_spec["variables"]) > 1:
        file_name += f"_{task_spec['variables'][1]}"
    file_name = file_name + ".json"
    
    with open(os.path.join("data", "results", "benchmark", file_name), "w") as f:
        json.dump(results, f, indent=4)

model_name = "llama3_8b_instruct"  # Example model name
tokenizer, model, is_instruct = load_model(model_name)

# evaluator(model_name, model, tokenizer, task_specs[0])

for i in range(len(task_specs)):
    evaluator(model_name, model, tokenizer, task_specs[i])
