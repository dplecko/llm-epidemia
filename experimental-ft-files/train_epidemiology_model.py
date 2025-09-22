#!/usr/bin/env python3
"""
Epidemiology Model Training Script
Uses underutilized GPUs, logs to wandb, saves checkpoints to /eb/local/shreyas
"""

import os
import json
import torch
import time
import psutil
import GPUtil
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
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EpidemiologyModelTrainer:
    """Trainer for epidemiology models with GPU optimization."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.selected_gpus = []
        
    def find_underutilized_gpus(self, num_gpus=2):
        """Find the most underutilized GPUs."""
        logger.info("🔍 Analyzing GPU utilization...")
        
        try:
            # Get GPU utilization
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                logger.warning("No GPUs found via GPUtil, falling back to CUDA devices")
                # Fallback to CUDA device count
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    self.selected_gpus = list(range(min(num_gpus, gpu_count)))
                    logger.info(f"Using GPUs: {self.selected_gpus}")
                    return self.selected_gpus
                else:
                    raise RuntimeError("No CUDA devices available")
            
            # Sort by utilization (ascending - least utilized first)
            gpus.sort(key=lambda x: x.load)
            
            # Select the least utilized GPUs
            self.selected_gpus = [gpu.id for gpu in gpus[:num_gpus]]
            
            logger.info(f"📊 GPU Analysis:")
            for gpu in gpus:
                status = "✅ SELECTED" if gpu.id in self.selected_gpus else "⏸️  Available"
                logger.info(f"   GPU {gpu.id}: {gpu.name} - Load: {gpu.load*100:.1f}% - Memory: {gpu.memoryUtil*100:.1f}% - {status}")
            
            return self.selected_gpus
            
        except Exception as e:
            logger.warning(f"Error analyzing GPUs: {e}")
            # Fallback to first available GPUs
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.selected_gpus = list(range(min(num_gpus, gpu_count)))
                logger.info(f"Using fallback GPUs: {self.selected_gpus}")
                return self.selected_gpus
            else:
                raise RuntimeError("No CUDA devices available")
    
    def setup_environment(self, gpu_ids):
        """Setup environment for selected GPUs."""
        logger.info(f"🔧 Setting up environment for GPUs: {gpu_ids}")
        
        # Set CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        # Verify GPU availability
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available with {torch.cuda.device_count()} visible devices")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"   Device {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            raise RuntimeError("CUDA not available after setup")
    
    def load_dataset(self, data_path="finetuning_setup/data/llama_formatted_data.json"):
        """Load the Llama-formatted dataset."""
        logger.info(f"📂 Loading dataset from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset format
        texts = [item['text'] for item in data]
        dataset = Dataset.from_dict({"text": texts})
        
        logger.info(f"✅ Loaded {len(dataset)} examples")
        return dataset
    
    def tokenize_dataset(self, dataset, max_length=512):
        """Tokenize the dataset for training."""
        logger.info("🔤 Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        logger.info("✅ Tokenization complete")
        return tokenized_dataset
    
    def setup_model_and_tokenizer(self, model_name="meta-llama/Llama-2-7b-hf"):
        """Setup model and tokenizer optimized for selected GPUs."""
        logger.info(f"🤖 Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimal settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for H100
            device_map="auto",
            trust_remote_code=True
        )
        
        # Setup LoRA with optimal configuration for H100
        lora_config = LoraConfig(
            r=32,  # Higher rank for better performance
            lora_alpha=64,  # Higher alpha
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
        
        return self.model, self.tokenizer
    
    def setup_wandb(self, project_name="llama-epidemiology", run_name=None):
        """Setup Weights & Biases logging."""
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"epidemiology-{timestamp}"
        
        logger.info(f"📊 Initializing wandb: {project_name}/{run_name}")
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model": "meta-llama/Llama-2-7b-hf",
                "method": "LoRA",
                "gpus": self.selected_gpus,
                "dataset": "NSDUH Persona Dataset",
                "samples": 27550
            }
        )
        
        return run_name
    
    def train_model(self, train_dataset, eval_dataset, output_dir="/eb/local/shreyas/llama-epidemiology"):
        """Train the model with comprehensive logging."""
        logger.info(f"🚀 Starting training...")
        logger.info(f"📁 Checkpoints will be saved to: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments optimized for H100
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,  # Large batch size for H100
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            learning_rate=3e-4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            run_name=f"epidemiology-gpu{self.selected_gpus[0]}",
            bf16=True,  # Use bfloat16 for H100
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_total_limit=3,  # Keep only 3 best checkpoints
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_first_step=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            include_inputs_for_metrics=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        start_time = time.time()
        logger.info("🏃‍♂️ Training started...")
        
        try:
            trainer.train()
            training_success = True
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            training_success = False
            raise
        
        end_time = time.time()
        training_time = (end_time - start_time) / 3600  # Convert to hours
        
        if training_success:
            logger.info(f"✅ Training completed successfully in {training_time:.2f} hours")
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Log final metrics
            final_metrics = trainer.evaluate()
            logger.info(f"📊 Final evaluation metrics: {final_metrics}")
            
            # Log to wandb
            wandb.log({
                "training_time_hours": training_time,
                "final_eval_loss": final_metrics.get("eval_loss", 0),
                "final_eval_perplexity": final_metrics.get("eval_perplexity", 0)
            })
            
            return trainer, final_metrics
        else:
            logger.error("❌ Training failed")
            return None, None
    
    def monitor_gpu_usage(self, duration=60):
        """Monitor GPU usage during training."""
        logger.info(f"📊 Monitoring GPU usage for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if gpu.id in self.selected_gpus:
                        logger.info(f"   GPU {gpu.id}: Load {gpu.load*100:.1f}%, Memory {gpu.memoryUtil*100:.1f}%, Temp {gpu.temperature}°C")
            except:
                pass
            time.sleep(10)
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up resources...")
        
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        torch.cuda.empty_cache()
        
        # Close wandb
        wandb.finish()
        
        logger.info("✅ Cleanup complete")

def main():
    """Main training function."""
    trainer = EpidemiologyModelTrainer()
    
    try:
        # Step 1: Find underutilized GPUs
        logger.info("="*60)
        logger.info("EPIDEMIOLOGY MODEL TRAINING")
        logger.info("="*60)
        
        selected_gpus = trainer.find_underutilized_gpus(num_gpus=2)
        trainer.selected_gpus = selected_gpus
        
        # Step 2: Setup environment
        trainer.setup_environment(selected_gpus)
        
        # Step 3: Setup wandb
        run_name = trainer.setup_wandb()
        
        # Step 4: Load dataset
        dataset = trainer.load_dataset()
        
        # Step 5: Setup model and tokenizer
        model, tokenizer = trainer.setup_model_and_tokenizer()
        
        # Step 6: Tokenize dataset
        tokenized_dataset = trainer.tokenize_dataset(dataset)
        
        # Step 7: Split dataset
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        logger.info(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Step 8: Train model
        trainer_instance, final_metrics = trainer.train_model(train_dataset, eval_dataset)
        
        if trainer_instance is not None:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"📁 Model saved to: /eb/local/shreyas/llama-epidemiology")
            logger.info(f"📊 Wandb run: {run_name}")
            
            # Show final results
            if final_metrics:
                logger.info("📈 Final Results:")
                for key, value in final_metrics.items():
                    logger.info(f"   {key}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()

if __name__ == "__main__":
    main()
