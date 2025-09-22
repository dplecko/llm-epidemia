#!/usr/bin/env python3
"""
Optimized Epidemiology Model Training Script
Uses /eb/local/shreyas for all storage to avoid disk quota issues
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

class OptimizedEpidemiologyTrainer:
    """Optimized trainer that uses /eb/local/shreyas for all storage."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.selected_gpus = []
        self.base_dir = Path("/local/eb/shreyas")
        self.cache_dir = self.base_dir / "huggingface_cache"
        self.model_dir = self.base_dir / "llama-epidemiology"
        self.data_dir = self.base_dir / "training_data"
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables to use our cache directory
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(self.cache_dir)
        
        logger.info(f"📁 Using cache directory: {self.cache_dir}")
        logger.info(f"📁 Model will be saved to: {self.model_dir}")
    
    def find_underutilized_gpus(self, num_gpus=2):
        """Find the most underutilized GPUs."""
        logger.info("🔍 Analyzing GPU utilization...")
        
        try:
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    self.selected_gpus = list(range(min(num_gpus, gpu_count)))
                    logger.info(f"Using GPUs: {self.selected_gpus}")
                    return self.selected_gpus
                else:
                    raise RuntimeError("No CUDA devices available")
            
            gpus.sort(key=lambda x: x.load)
            self.selected_gpus = [gpu.id for gpu in gpus[:num_gpus]]
            
            logger.info(f"📊 GPU Analysis:")
            for gpu in gpus:
                status = "✅ SELECTED" if gpu.id in self.selected_gpus else "⏸️  Available"
                logger.info(f"   GPU {gpu.id}: {gpu.name} - Load: {gpu.load*100:.1f}% - Memory: {gpu.memoryUtil*100:.1f}% - {status}")
            
            return self.selected_gpus
            
        except Exception as e:
            logger.warning(f"Error analyzing GPUs: {e}")
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
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available with {torch.cuda.device_count()} visible devices")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"   Device {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            raise RuntimeError("CUDA not available after setup")
    
    def load_dataset(self, source_data_path="finetuning_setup/data/llama_formatted_data.json"):
        """Load and copy dataset to /eb/local/shreyas."""
        logger.info(f"📂 Loading dataset from {source_data_path}")
        
        if not os.path.exists(source_data_path):
            raise FileNotFoundError(f"Source dataset not found: {source_data_path}")
        
        # Copy dataset to our data directory
        target_data_path = self.data_dir / "llama_formatted_data.json"
        logger.info(f"📋 Copying dataset to {target_data_path}")
        
        import shutil
        shutil.copy2(source_data_path, target_data_path)
        
        with open(target_data_path, 'r') as f:
            data = json.load(f)
        
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
        """Setup model and tokenizer with optimized settings."""
        logger.info(f"🤖 Loading model: {model_name}")
        logger.info(f"📁 Cache directory: {self.cache_dir}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(self.cache_dir),
            low_cpu_mem_usage=True  # Reduce memory usage during loading
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=16,  # Reduced rank to save memory
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
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
                "samples": 27550,
                "cache_dir": str(self.cache_dir),
                "model_dir": str(self.model_dir)
            }
        )
        
        return run_name
    
    def train_model(self, train_dataset, eval_dataset):
        """Train the model with comprehensive logging."""
        logger.info(f"🚀 Starting training...")
        logger.info(f"📁 Checkpoints will be saved to: {self.model_dir}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments optimized for H100 with memory constraints
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            per_device_train_batch_size=4,  # Reduced batch size
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Increased to maintain effective batch size
            num_train_epochs=2,  # Reduced epochs for faster training
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
            run_name=f"epidemiology-gpu{self.selected_gpus[0]}",
            bf16=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,  # Reduced workers
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_total_limit=2,  # Keep only 2 checkpoints
            logging_dir=str(self.model_dir / "logs"),
            logging_strategy="steps",
            logging_first_step=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            include_inputs_for_metrics=False,
            dataloader_drop_last=True  # Drop incomplete batches
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
        training_time = (end_time - start_time) / 3600
        
        if training_success:
            logger.info(f"✅ Training completed successfully in {training_time:.2f} hours")
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(str(self.model_dir))
            
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
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up resources...")
        
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        torch.cuda.empty_cache()
        wandb.finish()
        
        logger.info("✅ Cleanup complete")

def main():
    """Main training function."""
    trainer = OptimizedEpidemiologyTrainer()
    
    try:
        logger.info("="*60)
        logger.info("OPTIMIZED EPIDEMIOLOGY MODEL TRAINING")
        logger.info("="*60)
        
        # Step 1: Find underutilized GPUs
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
            logger.info(f"📁 Model saved to: {trainer.model_dir}")
            logger.info(f"📊 Wandb run: {run_name}")
            
            if final_metrics:
                logger.info("📈 Final Results:")
                for key, value in final_metrics.items():
                    logger.info(f"   {key}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()
