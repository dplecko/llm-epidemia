#!/usr/bin/env python3
"""
Training script with fixed tokenization and data collator approach
"""

import os
import json
import torch
import time
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import wandb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedDataCollator:
    """Custom data collator that handles the tokenized data properly."""
    
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, features):
        # Extract input_ids and attention_mask from features
        batch = {}
        
        # Get the maximum length in this batch
        max_length = max(len(f['input_ids']) for f in features)
        
        # Pad sequences to the same length
        input_ids = []
        attention_masks = []
        
        for feature in features:
            input_id = feature['input_ids']
            attention_mask = feature['attention_mask']
            
            # Pad to max_length
            padding_length = max_length - len(input_id)
            if padding_length > 0:
                input_id = input_id + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['attention_mask'] = torch.tensor(attention_masks, dtype=torch.long)
        
        # Create labels for causal language modeling
        if self.mlm:
            # For MLM, labels are the same as input_ids
            batch['labels'] = batch['input_ids'].clone()
        else:
            # For causal LM, labels are the same as input_ids (shifted by 1 in the model)
            batch['labels'] = batch['input_ids'].clone()
        
        return batch

def main():
    """Fixed training function with proper data collator."""
    try:
        logger.info("="*60)
        logger.info("FIXED DATA COLLATOR TRAINING")
        logger.info("="*60)
        
        # Setup directories
        base_dir = Path("/local/eb/shreyas")
        cache_dir = base_dir / "huggingface_cache"
        model_dir = base_dir / "llama-epidemiology"
        
        # Create directories
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
        
        logger.info(f"📁 Using cache directory: {cache_dir}")
        logger.info(f"📁 Model will be saved to: {model_dir}")
        
        # Setup wandb
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"epidemiology-{timestamp}"
        
        logger.info(f"📊 Initializing wandb: llama-epidemiology/{run_name}")
        
        wandb.init(
            project="llama-epidemiology",
            name=run_name,
            config={
                "model": "meta-llama/Llama-2-7b-hf",
                "method": "LoRA",
                "dataset": "NSDUH Persona Dataset",
                "samples": 27550,
                "cache_dir": str(cache_dir),
                "model_dir": str(model_dir)
            }
        )
        
        # Load dataset
        source_data_path = "finetuning_setup/data/llama_formatted_data.json"
        logger.info(f"📂 Loading dataset from {source_data_path}")
        
        with open(source_data_path, 'r') as f:
            data = json.load(f)
        
        # Use all data
        texts = [item['text'] for item in data]
        dataset = Dataset.from_dict({"text": texts})
        
        logger.info(f"✅ Loaded {len(dataset)} examples")
        
        # Load model and tokenizer
        model_name = "meta-llama/Llama-2-7b-hf"
        logger.info(f"🤖 Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            low_cpu_mem_usage=True
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
        
        # Tokenize dataset with proper approach
        logger.info("🔤 Tokenizing dataset...")
        
        def tokenize_function(examples):
            # Tokenize each text individually to avoid batching issues
            tokenized = []
            for text in examples['text']:
                tokens = tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=512,
                    return_tensors=None
                )
                tokenized.append(tokens)
            
            # Convert to the format expected by the dataset
            return {
                'input_ids': [t['input_ids'] for t in tokenized],
                'attention_mask': [t['attention_mask'] for t in tokenized]
            }
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=1)
        logger.info("✅ Tokenization complete")
        
        # Split dataset
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        logger.info(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Setup training with custom data collator
        data_collator = FixedDataCollator(tokenizer=tokenizer, mlm=False)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(model_dir),
            per_device_train_batch_size=2,  # Smaller batch size
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Increase to maintain effective batch size
            num_train_epochs=2,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            bf16=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_total_limit=2,
            logging_dir=str(model_dir / "logs"),
            logging_strategy="steps",
            logging_first_step=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            include_inputs_for_metrics=False,
            dataloader_drop_last=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Start training
        start_time = time.time()
        logger.info("🏃‍♂️ Training started...")
        logger.info(f"📁 Checkpoints will be saved to: {model_dir}")
        
        # Check GPU utilization before training
        if torch.cuda.is_available():
            logger.info(f"🔍 GPU status before training:")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                logger.info(f"   GPU {i}: {gpu_name} - Memory: {memory_used:.1f}GB / {gpu_memory:.1f}GB")
        
        try:
            trainer.train()
            training_success = True
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            training_success = False
            raise
        
        end_time = time.time()
        training_time = (end_time - start_time) / 3600  # in hours
        
        if training_success:
            logger.info(f"✅ Training completed successfully in {training_time:.2f} hours")
            
            # Save final model
            trainer.save_model()
            tokenizer.save_pretrained(str(model_dir))
            
            # Log final metrics
            final_metrics = trainer.evaluate()
            logger.info(f"📊 Final evaluation metrics: {final_metrics}")
            
            # Log to wandb
            wandb.log({
                "training_time_hours": training_time,
                "final_eval_loss": final_metrics.get("eval_loss", 0),
                "final_eval_perplexity": final_metrics.get("eval_perplexity", 0)
            })
            
            logger.info("🎉 Training completed successfully!")
            logger.info(f"📁 Model saved to: {model_dir}")
            logger.info(f"📊 Wandb run: {run_name}")
            
            if final_metrics:
                logger.info("📈 Final Results:")
                for key, value in final_metrics.items():
                    logger.info(f"   {key}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up resources...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        wandb.finish()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()
