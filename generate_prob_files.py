#!/usr/bin/env python3
"""
Generate PROB files for fine-tuned BRFSS models
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Add workspace to path
sys.path.append(os.path.abspath("workspace"))
sys.path.append(os.path.abspath("workspace/utils"))

from workspace.tasks.tasks_brfss import tasks_brfss, tasks_brfss_hd
from workspace.utils.hd_helpers import gen_prob_lvls
from workspace.utils.helpers import task_to_filename, dat_name_clean
from workspace.utils.extract_helpers import compress_vals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinetunedModelWrapper:
    """Wrapper for fine-tuned models to generate PROB predictions."""
    
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        logger.info(f"Loading {self.model_name} from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=dtype, 
            device_map="auto", 
            attn_implementation="eager"
        )
        logger.info(f"✅ {self.model_name} loaded successfully")
    
    def predict_prob(self, prompt, levels, n_mc=128):
        """Generate probability distribution predictions."""
        import torch
        from transformers import GenerationConfig
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=n_mc,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_config=GenerationConfig(
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            )
        
        # Decode responses
        responses = []
        for output in outputs:
            # Get only the new tokens (excluding the input prompt)
            new_tokens = output[inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(response)
        
        # Count occurrences of each probability level
        level_counts = {}
        for response in responses:
            for level in levels:
                if level.lower() in response.lower():
                    level_counts[level] = level_counts.get(level, 0) + 1
                    break
        
        # Convert to probabilities
        total_responses = len(responses)
        if total_responses > 0:
            weights = [level_counts.get(level, 0) / total_responses for level in levels]
        else:
            weights = [1.0 / len(levels)] * len(levels)
        
        return levels, weights

def generate_prob_json_for_task(model_name: str, model, task: dict, output_dir: str) -> str | None:
    """Generate PROB JSON for a single task."""
    dataset_path = task["dataset"]
    if not Path(dataset_path).exists():
        logger.error(f"Missing dataset: {dataset_path}")
        return None

    data = pd.read_parquet(dataset_path)
    if data.empty:
        logger.error("Source dataset is empty; skipping")
        return None

    # Handle both 1D and HD task formats
    if "variables" in task:
        # 1D task format
        out_var = task["variables"][0]
        if len(task["variables"]) == 2:
            cond_var = task["variables"][1]
            cond_values = data[cond_var].dropna().unique().tolist()
        else:
            cond_var = None
            cond_values = [None]
        
        # Build prompts for 1D tasks
        prompts = []
        conditions = []
        for cv in cond_values:
            if cond_var is None:
                prompts.append(task["prompt"])
                conditions.append("All")
            else:
                prompts.append(task["prompt"].format(cv))
                conditions.append(cv.tolist() if hasattr(cv, "tolist") else cv)
                
    elif "v_out" in task:
        # HD task format
        out_var = task["v_out"]
        cond_vars = task["v_cond"]
        
        # Build prompts for HD tasks
        prompts = []
        conditions = []
        
        # Get unique combinations of conditioning variables
        if cond_vars:
            cond_combinations = data[cond_vars].drop_duplicates().values
            for combo in cond_combinations:
                # Create a simple prompt for HD tasks
                cond_str = " and ".join([f"{var}={val}" for var, val in zip(cond_vars, combo)])
                prompt = f"What is the probability that a person with {cond_str} has {out_var}?"
                prompts.append(prompt)
                conditions.append(cond_str)
        else:
            prompt = f"What is the probability that a person has {out_var}?"
            prompts.append(prompt)
            conditions.append("All")
    else:
        logger.error(f"Unknown task format: {task}")
        return None

    # Get 21-bin probability levels
    levels = gen_prob_lvls()

    # Evaluate each prompt
    results = []
    for i, (prompt, cond) in enumerate(zip(prompts, conditions)):
        logger.info(f"  Processing condition: {cond}")
        
        # Get model probability predictions
        model_vals, model_weights = model.predict_prob(prompt, levels, n_mc=128)
        
        # Get ground truth
        if "variables" in task:
            # 1D task
            if cond_var is None:
                mask = pd.Series([True] * len(data))
            else:
                mask = (data[cond_var] == cond_values[i])
        else:
            # HD task
            if cond_vars:
                mask = pd.Series([True] * len(data))
                for var, val in zip(cond_vars, cond_combinations[i]):
                    mask &= (data[var] == val)
            else:
                mask = pd.Series([True] * len(data))

        true_vals = data.loc[mask, out_var].tolist()
        # Determine weights if available
        if "weight" in data.columns:
            true_w = data.loc[mask, "weight"].tolist()
        else:
            true_w = [1.0] * len(true_vals)

        # Compress truth to value/weight lists
        true_vals_c, true_w_c = compress_vals(true_vals, true_w)

        results.append({
            "condition": cond,
            "true_vals": true_vals_c,
            "true_weights": true_w_c,
            "n_data": int(len(data.loc[mask])),
            "total_weight": float(sum(true_w)),
            "model_vals": model_vals,
            "model_weights": model_weights,
        })

    # Write JSON
    file_name = task_to_filename(model_name, task)
    file_name = f"PROB_{file_name}"
    out_path = Path(output_dir) / f"{file_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return str(out_path)

def main():
    """Generate PROB files for both fine-tuned models."""
    
    logger.info("="*60)
    logger.info("GENERATING PROB FILES FOR FINE-TUNED MODELS")
    logger.info("="*60)
    
    # Define models and their paths
    models = {
        "llama3_8b_instruct_brfss_finetuned": "/local/eb/shreyas/models/llama3_8b_instruct_brfss_finetuned/best",
        "mistral_7b_instruct_brfss_finetuned": "/local/eb/shreyas/models/mistral_7b_instruct_brfss_finetuned/best"
    }
    
    # Filter tasks to only include PROB-supported tasks
    # All BRFSS 1D tasks are PROB tasks
    prob_1d_tasks = tasks_brfss
    
    # All BRFSS HD tasks are PROB tasks
    prob_hd_tasks = tasks_brfss_hd
    
    logger.info(f"Found {len(prob_1d_tasks)} PROB-supported 1D tasks")
    logger.info(f"Found {len(prob_hd_tasks)} PROB-supported HD tasks")
    
    # Process each model
    for model_name, model_path in models.items():
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            continue
            
        logger.info(f"🤖 Processing {model_name}")
        
        # Load model
        model = FinetunedModelWrapper(model_path, model_name)
        
        # Generate 1D PROB files
        logger.info("📊 Generating 1D PROB files...")
        output_dir_1d = "/local/eb/shreyas/benchmark-brfss-finetuned"
        successful_1d = 0
        for i, task in enumerate(tqdm(prob_1d_tasks, desc=f"1D PROB tasks for {model_name}")):
            try:
                result = generate_prob_json_for_task(
                    model_name=model_name,
                    model=model,
                    task=task,
                    output_dir=output_dir_1d
                )
                if result:
                    successful_1d += 1
                    logger.info(f"  ✅ 1D PROB task {i+1} completed")
                else:
                    logger.info(f"  ⏭️ 1D PROB task {i+1} skipped")
            except Exception as e:
                logger.error(f"  ❌ 1D PROB task {i+1} failed: {e}")
        
        # Generate HD PROB files
        logger.info("📊 Generating HD PROB files...")
        output_dir_hd = "/local/eb/shreyas/benchmark-brfss-finetuned-hd"
        successful_hd = 0
        for i, task in enumerate(tqdm(prob_hd_tasks, desc=f"HD PROB tasks for {model_name}")):
            try:
                result = generate_prob_json_for_task(
                    model_name=model_name,
                    model=model,
                    task=task,
                    output_dir=output_dir_hd
                )
                if result:
                    successful_hd += 1
                    logger.info(f"  ✅ HD PROB task {i+1} completed")
                else:
                    logger.info(f"  ⏭️ HD PROB task {i+1} skipped")
            except Exception as e:
                logger.error(f"  ❌ HD PROB task {i+1} failed: {e}")
        
        logger.info(f"📊 {model_name} PROB Summary:")
        logger.info(f"  ✅ 1D PROB tasks: {successful_1d}/{len(prob_1d_tasks)}")
        logger.info(f"  ✅ HD PROB tasks: {successful_hd}/{len(prob_hd_tasks)}")
    
    logger.info("🎉 PROB file generation completed!")

if __name__ == "__main__":
    main()
