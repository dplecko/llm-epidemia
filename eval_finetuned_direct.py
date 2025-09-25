#!/usr/bin/env python3
"""
Direct evaluation of BRFSS fine-tuned models using extract.py approach
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import logging
from pathlib import Path

# Add workspace to path
sys.path.append(os.path.abspath("workspace"))
sys.path.append(os.path.abspath("workspace/utils"))

from workspace.tasks.tasks_brfss import tasks_brfss
from workspace.utils.helpers import task_to_filename, dat_name_clean
from workspace.utils.extract_helpers import compress_vals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinetunedModelWrapper:
    """Wrapper for fine-tuned model to match the expected interface."""
    
    def __init__(self, model_path, model_name):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
    def predict(self, prompt, levels, n_mc=128, max_batch_size=128):
        """Single prompt prediction using the fine-tuned model."""
        import torch
        from transformers import GenerationConfig
        
        # Flatten levels if they're nested
        if levels and isinstance(levels[0], list):
            flat_levels = [level[0] for level in levels]
        else:
            flat_levels = levels
        
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
        
        # Count occurrences of each level
        level_counts = {}
        for response in responses:
            for level in flat_levels:
                if level.lower() in response.lower():
                    level_counts[level] = level_counts.get(level, 0) + 1
                    break
        
        # Convert to indices and weights
        total_responses = len(responses)
        if total_responses > 0:
            weights = [level_counts.get(level, 0) / total_responses for level in flat_levels]
        else:
            weights = [1.0 / len(flat_levels)] * len(flat_levels)
        
        # Return indices and weights
        indices = list(range(len(flat_levels)))
        return indices, weights

def generate_json_for_task(model_name: str, model, task: dict, output_dir: str, prob: bool = False) -> str | None:
    """
    Generate evaluation JSON for a single BRFSS task using the fine-tuned model.
    """
    dataset_path = task["dataset"]
    if not Path(dataset_path).exists():
        logger.error(f"Missing dataset: {dataset_path}")
        return None

    data = pd.read_parquet(dataset_path)
    if data.empty:
        logger.error("Source dataset is empty; skipping")
        return None

    # Determine conditioning values
    if len(task["variables"]) == 2:
        cond_var = task["variables"][1]
        cond_values = data[cond_var].dropna().unique().tolist()
    else:
        cond_var = None
        cond_values = [None]

    # Build prompts
    dataset_name = dat_name_clean(dataset_path)
    prompts = []
    conditions = []
    for cv in cond_values:
        if cond_var is None:
            prompts.append(task["prompt"])
            conditions.append("All")
        else:
            prompts.append(task["prompt"].format(cv))
            conditions.append(cv.tolist() if hasattr(cv, "tolist") else cv)

    # Get levels for the output variable
    out_var = task["variables"][0]
    if prob:
        # For PROB mode, use 21-bin probability levels
        from workspace.utils.hd_helpers import gen_prob_lvls
        levels = gen_prob_lvls()
    else:
        levels = data[out_var].dropna().unique().tolist()
        levels = [[level] for level in levels]  # Convert to expected format

    # Evaluate each prompt
    results = []
    for i, (prompt, cond) in enumerate(zip(prompts, conditions)):
        logger.info(f"  Processing condition: {cond}")
        
        # Get model predictions
        model_vals, model_weights = model.predict(prompt, levels, n_mc=128, prob=prob)
        
        # Get ground truth
        if cond_var is None:
            mask = pd.Series([True] * len(data))
        else:
            mask = (data[cond_var] == cond_values[i])

        true_vals = data.loc[mask, out_var].tolist()
        # Determine weights if available
        if "weight" in data.columns:
            true_w = data.loc[mask, "weight"].tolist()
        else:
            true_w = [1.0] * len(true_vals)

        # Compress truth to value/weight lists
        true_vals_c, true_w_c = compress_vals(true_vals, true_w)

        # Convert model predictions to the expected format
        if prob:
            # For PROB mode, use the full distribution
            model_vals_c = levels
            model_w_c = model_weights
        else:
            # For regular mode, use the sampled values
            model_vals_c = [levels[idx][0] for idx in model_vals]
            model_w_c = model_weights

        results.append({
            "condition": cond,
            "true_vals": true_vals_c,
            "true_weights": true_w_c,
            "n_data": int(len(data.loc[mask])),
            "total_weight": float(sum(true_w)),
            "model_vals": model_vals_c,
            "model_weights": model_w_c,
        })

    # Write JSON
    file_name = task_to_filename(model_name, task)
    if prob:
        file_name = f"PROB_{file_name}"
    out_path = Path(output_dir) / f"{file_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return str(out_path)

def main():
    """Run direct evaluation on BRFSS fine-tuned models."""
    
    logger.info("="*60)
    logger.info("DIRECT BRFSS FINE-TUNED MODELS EVALUATION")
    logger.info("="*60)
    
    # Model configurations
    models = [
        {
            "name": "llama3_8b_instruct_brfss_finetuned",
            "path": "/local/eb/shreyas/models/llama3_8b_instruct_brfss_finetuned/best"
        },
        {
            "name": "mistral_7b_instruct_brfss_finetuned", 
            "path": "/local/eb/shreyas/models/mistral_7b_instruct_brfss_finetuned/best"
        }
    ]
    
    # Create output directory
    output_dir = "/local/eb/shreyas/benchmark-brfss-finetuned"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_config in models:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        logger.info(f"🤖 Processing model: {model_name}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"❌ Fine-tuned model not found at: {model_path}")
            continue
        
        # Load the fine-tuned model
        logger.info(f"  Loading model from: {model_path}")
        try:
            model = FinetunedModelWrapper(model_path, model_name)
            logger.info(f"  ✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"  ❌ Error loading model: {e}")
            continue
        
        # Run evaluation on all BRFSS tasks (both regular and PROB modes)
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task in enumerate(tqdm(tasks_brfss, desc=f"Processing {model_name}")):
            logger.info(f"🔄 Processing task {i+1}/{len(tasks_brfss)}: {task.get('name', f'Task {i+1}')}")
            
            # Check if this task supports PROB mode (diabetes and high_bp)
            prob_supported = task.get('variables', [''])[0] in ['diabetes', 'high_bp']
            
            # Run regular mode
            try:
                result = generate_json_for_task(
                    model_name=model_name,
                    model=model,
                    task=task,
                    output_dir=output_dir,
                    prob=False
                )
                
                if result is not None:
                    successful_tasks += 1
                    logger.info(f"  ✅ Task {i+1} (regular) completed successfully")
                else:
                    logger.info(f"  ⏭️ Task {i+1} (regular) skipped")
                    successful_tasks += 1
                    
            except Exception as e:
                failed_tasks += 1
                logger.error(f"  ❌ Task {i+1} (regular) failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Run PROB mode if supported
            if prob_supported:
                try:
                    result = generate_json_for_task(
                        model_name=model_name,
                        model=model,
                        task=task,
                        output_dir=output_dir,
                        prob=True
                    )
                    
                    if result is not None:
                        successful_tasks += 1
                        logger.info(f"  ✅ Task {i+1} (PROB) completed successfully")
                    else:
                        logger.info(f"  ⏭️ Task {i+1} (PROB) skipped")
                        successful_tasks += 1
                        
                except Exception as e:
                    failed_tasks += 1
                    logger.error(f"  ❌ Task {i+1} (PROB) failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Summary for this model
        logger.info(f"📊 {model_name} Summary:")
        logger.info(f"  ✅ Successful tasks: {successful_tasks}")
        logger.info(f"  ❌ Failed tasks: {failed_tasks}")
        logger.info(f"  📊 Total tasks: {len(tasks_brfss)}")
    
    logger.info("🎉 Direct evaluation completed!")
    logger.info(f"📁 Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
