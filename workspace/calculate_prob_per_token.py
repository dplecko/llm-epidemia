"""
Calculates the average log probability per token for a given text string. 
Used for data memorization analysis.
"""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import json
from tqdm import tqdm
from utils.hd_helpers import fit_lgbm, promptify, gen_prob_lvls
from utils.helpers import load_dts
from common import *
from extract import get_ground_truth
from tqdm import tqdm


def calculate_log_prob_per_token(model, tokenizer, text):
    """
    Calculates the average log probability per token for a given text string.

    Args:
        model: A HuggingFace PreTrainedModel
        tokenizer: A HuggingFace PreTrainedTokenizer
        text: The input string (question/prompt) to analyze.

    Returns:
        float: The average log probability per token (a negative number).
               Closer to 0 means higher probability (more likely memorized).
               More negative means lower probability (more surprising to the model).
    """
    
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # We use torch.no_grad() because we don't need gradients
    with torch.no_grad():
        # Passing 'labels' triggers the model to compute the loss automatically.
        # The model internally shifts the tokens so it predicts the next token.
        outputs = model(**inputs, labels=inputs["input_ids"])

    # CrossEntropyLoss is equivalent to the Average Negative Log Likelihood.
    # So: Loss = - (Average Log Probability)
    # Average Log Probability = - Loss
    nll = outputs.loss.item()
    return -nll


def prepare_questions(task_spec, model, prob=False):    
    # Step 1: determine if query is marginal or conditional
    if "v_cond" in task_spec:
        ttyp = "hd"
    elif len(task_spec["variables"]) == 1:
        ttyp = "marginal"
    elif len(task_spec["variables"]) == 2:
        ttyp = "conditional"
    
    data = load_dts(task_spec, None)

    if ttyp in ["hd"]:
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
        answers, _ = model.prepare_answers(levels)
        prompt = task_spec["prompt"] + answers
        return {prompt: calculate_log_prob_per_token(model.model, model.tokenizer, prompt)}

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

            res = {}
            if isinstance(prompt, list):
                for p in prompt:
                    answers, _ = model.prepare_answers(levels)
                    proc_p = p + answers
                    res[proc_p] = calculate_log_prob_per_token(model.model, model.tokenizer, proc_p)
            else:
                answers, _ = model.prepare_answers(levels)
                proc_p = prompt + answers
                res[proc_p] = calculate_log_prob_per_token(model.model, model.tokenizer, proc_p)
            return res
 
    elif ttyp == "hd":
        cond_vars = task_spec["v_cond"]
        out_var = task_spec["v_out"]
        
        # fit lightgbm model to get conditional probabilities (ground truth)
        preds = fit_lgbm(data, out_var, cond_vars, wgh_col="weight")
        data["lgbm_pred"] = preds
        
        # get all combinations
        cond_df = data[cond_vars].drop_duplicates().reset_index(drop=True)
        cond_df["llm_pred"] = np.nan
        dataset_name = task_spec['dataset'].split('/')[-1].split('.')[0]
        prompts = [
            promptify(out_var, cond_vars, row, dataset_name, prob=prob)  # type: ignore[name‑defined]
            for _, row in cond_df.iterrows()
        ]
        res = {}
        if isinstance(prompts, list):
            for p in prompts:
                answers, _ = model.prepare_answers(levels)
                proc_p = p + answers
                res[proc_p] = calculate_log_prob_per_token(model.model, model.tokenizer, proc_p)
        else:
            answers, _ = model.prepare_answers(levels)
            proc_p = p + answers
            res[proc_p] = calculate_log_prob_per_token(model.model, model.tokenizer, proc_p)
        return res


if __name__ == "__main__":
    model_names = ["llama3_8b_instruct", "phi4"]
    for model_name in model_names:
        model = load_model(model_name)
        
        results = {}
        for task in tqdm(task_specs):
            results.update(prepare_questions(task, model, prob=False).items())
            
        with open(f"log_probs_{model_name}_lowdim.json", "w") as f:
            json.dump(results, f)
        
        results_hd = {}
        for task_hd in tqdm(task_specs_hd):
            results_hd.update(prepare_questions(task_hd, model, prob=False).items())
            
        with open(f"log_probs_{model_name}_highdim.json", "w") as f:
            json.dump(results_hd, f)