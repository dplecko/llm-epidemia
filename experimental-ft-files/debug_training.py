#!/usr/bin/env python3
"""
Debug script to understand where training hangs
"""

import os
import json
import torch
import time
import psutil
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

def check_memory():
    """Check current memory usage."""
    logger.info("🔍 Checking memory usage...")
    
    # System memory
    memory = psutil.virtual_memory()
    logger.info(f"   System RAM: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            logger.info(f"   GPU {i} ({gpu_name}): {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved, {gpu_memory:.1f}GB total")

def test_model_loading():
    """Test model loading step by step."""
    logger.info("="*60)
    logger.info("TESTING MODEL LOADING STEP BY STEP")
    logger.info("="*60)
    
    # Setup directories
    base_dir = Path("/local/eb/shreyas")
    cache_dir = base_dir / "huggingface_cache"
    model_dir = base_dir / "llama-epidemiology-debug"
    
    # Create directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    
    logger.info(f"📁 Using cache directory: {cache_dir}")
    logger.info(f"📁 Model will be saved to: {model_dir}")
    
    # Check memory before loading
    check_memory()
    
    # Step 1: Load tokenizer
    logger.info("🔤 Step 1: Loading tokenizer...")
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir=str(cache_dir)
        )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✅ Tokenizer loaded in {time.time() - start_time:.2f}s")
        check_memory()
    except Exception as e:
        logger.error(f"❌ Tokenizer loading failed: {e}")
        return
    
    # Step 2: Load model
    logger.info("🤖 Step 2: Loading model...")
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            low_cpu_mem_usage=True
        )
        logger.info(f"✅ Model loaded in {time.time() - start_time:.2f}s")
        check_memory()
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return
    
    # Step 3: Setup LoRA
    logger.info("🔧 Step 3: Setting up LoRA...")
    start_time = time.time()
    
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        logger.info(f"✅ LoRA setup complete in {time.time() - start_time:.2f}s")
        check_memory()
    except Exception as e:
        logger.error(f"❌ LoRA setup failed: {e}")
        return
    
    # Step 4: Load dataset
    logger.info("📂 Step 4: Loading dataset...")
    start_time = time.time()
    
    try:
        source_data_path = "finetuning_setup/data/llama_formatted_data.json"
        with open(source_data_path, 'r') as f:
            data = json.load(f)
        
        # Use only first 100 examples for testing
        texts = [item['text'] for item in data[:100]]
        dataset = Dataset.from_dict({"text": texts})
        
        logger.info(f"✅ Dataset loaded in {time.time() - start_time:.2f}s: {len(dataset)} examples")
        check_memory()
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return
    
    # Step 5: Tokenize dataset
    logger.info("🔤 Step 5: Tokenizing dataset...")
    start_time = time.time()
    
    try:
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        logger.info(f"✅ Tokenization complete in {time.time() - start_time:.2f}s")
        check_memory()
    except Exception as e:
        logger.error(f"❌ Tokenization failed: {e}")
        return
    
    # Step 6: Test data collator
    logger.info("🔧 Step 6: Testing data collator...")
    start_time = time.time()
    
    try:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Test with a small batch
        sample_batch = tokenized_dataset.select(range(2))
        test_batch = data_collator([sample_batch[i] for i in range(2)])
        logger.info(f"✅ Data collator test passed in {time.time() - start_time:.2f}s")
        check_memory()
    except Exception as e:
        logger.error(f"❌ Data collator test failed: {e}")
        return
    
    # Step 7: Test training arguments
    logger.info("⚙️ Step 7: Testing training arguments...")
    start_time = time.time()
    
    try:
        training_args = TrainingArguments(
            output_dir=str(model_dir),
            per_device_train_batch_size=1,  # Very small for testing
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_steps=1,
            save_steps=10,
            eval_steps=10,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to="none",  # Disable wandb for testing
            bf16=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_total_limit=1,
            logging_dir=str(model_dir / "logs"),
            logging_strategy="steps",
            logging_first_step=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            include_inputs_for_metrics=False,
            dataloader_drop_last=True
        )
        logger.info(f"✅ Training arguments created in {time.time() - start_time:.2f}s")
        check_memory()
    except Exception as e:
        logger.error(f"❌ Training arguments creation failed: {e}")
        return
    
    # Step 8: Test trainer creation
    logger.info("🏃 Step 8: Testing trainer creation...")
    start_time = time.time()
    
    try:
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        logger.info(f"✅ Trainer created in {time.time() - start_time:.2f}s")
        check_memory()
    except Exception as e:
        logger.error(f"❌ Trainer creation failed: {e}")
        return
    
    # Step 9: Test one training step
    logger.info("🚀 Step 9: Testing one training step...")
    start_time = time.time()
    
    try:
        # Get one batch
        train_dataloader = trainer.get_train_dataloader()
        batch = next(iter(train_dataloader))
        logger.info(f"✅ Batch created in {time.time() - start_time:.2f}s")
        logger.info(f"   Batch keys: {batch.keys()}")
        logger.info(f"   Input shape: {batch['input_ids'].shape}")
        check_memory()
        
        # Test forward pass
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)
        logger.info(f"✅ Forward pass completed in {time.time() - start_time:.2f}s")
        logger.info(f"   Output logits shape: {outputs.logits.shape}")
        check_memory()
        
    except Exception as e:
        logger.error(f"❌ Training step test failed: {e}")
        return
    
    logger.info("🎉 All tests passed! Training should work.")
    
    # Cleanup
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    check_memory()

def main():
    """Main debug function."""
    try:
        logger.info("="*60)
        logger.info("TRAINING DEBUG SCRIPT")
        logger.info("="*60)
        
        # Initial memory check
        check_memory()
        
        # Test model loading
        test_model_loading()
        
    except Exception as e:
        logger.error(f"❌ Debug failed with error: {e}")
        raise
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        logger.info("🧹 Cleanup complete")

if __name__ == "__main__":
    main()
