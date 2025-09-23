#!/usr/bin/env python3
"""
LoRA Fine-tuning script for NSDUH Synthetic Gemma Dataset (Mistral 7B Instruct model)
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
from datasets import load_from_disk
import wandb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("="*60)
        logger.info("NSDUH SYNTHETIC GEMMA DATA TRAINING (Mistral 7B Instruct)")
        logger.info("="*60)

        base_dir = Path("/home/shreyas/llm-epidemia")
        cache_dir = base_dir / "huggingface_cache"
        model_dir = base_dir / "models" / "nsduh_synth_mistral_instruct"
        data_dir = base_dir / "external_data" / "data" / "nsduh_synth_gemma"

        cache_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

        logger.info(f"📁 Using cache directory: {cache_dir}")
        logger.info(f"📁 Model will be saved to: {model_dir}")
        logger.info(f"📁 Loading data from: {data_dir}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"nsduh-synth-mistral-instruct-{timestamp}"
        logger.info(f"📊 Initializing wandb: llm-epidemia/{run_name}")

        wandb.init(
            project="llm-epidemia",
            name=run_name,
            config={
                "model": "local-mistral-7b-instruct",
                "model_path": "/local/eb/dp3144/mistral_7b_instruct",
                "method": "LoRA",
                "dataset": "NSDUH Synthetic Gemma",
                "train_samples": 2000,
                "val_samples": 500,
                "total_samples": 2500,
                "cache_dir": str(cache_dir),
                "model_dir": str(model_dir),
                "data_dir": str(data_dir)
            }
        )

        logger.info(f"📂 Loading dataset from {data_dir}")
        dataset = load_from_disk(str(data_dir))
        logger.info(f"✅ Loaded dataset with splits: {list(dataset.keys())}")
        logger.info(f"📊 Train samples: {len(dataset['train'])}")
        logger.info(f"📊 Validation samples: {len(dataset['validation'])}")

        model_path = "/local/eb/dp3144/mistral_7b_instruct"
        logger.info(f"🤖 Loading local Mistral-7B Instruct model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=str(cache_dir)
        )
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            low_cpu_mem_usage=True
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.train()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

        logger.info("🔤 Tokenizing dataset...")

        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']
            }

        train_dataset = dataset['train'].map(tokenize_function, batched=True, remove_columns=['text'])
        eval_dataset = dataset['validation'].map(tokenize_function, batched=True, remove_columns=['text'])

        logger.info("✅ Tokenization complete")
        logger.info(f"📊 Tokenized train samples: {len(train_dataset)}")
        logger.info(f"📊 Tokenized eval samples: {len(eval_dataset)}")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir=str(model_dir),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
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
            gradient_checkpointing=False,
            optim="adamw_torch_fused",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_total_limit=3,
            logging_dir=str(model_dir / "logs"),
            logging_strategy="steps",
            logging_first_step=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            include_inputs_for_metrics=False,
            dataloader_drop_last=True,
            save_safetensors=True,
            push_to_hub=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        start_time = time.time()
        logger.info("🏃‍♂️ Training started...")
        logger.info(f"📁 Checkpoints will be saved to: {model_dir}")

        trainer.train()

        training_time = (time.time() - start_time) / 3600
        logger.info(f"✅ Training completed successfully in {training_time:.2f} hours")

        trainer.save_model()
        tokenizer.save_pretrained(str(model_dir))
        final_metrics = trainer.evaluate()
        logger.info(f"📊 Final evaluation metrics: {final_metrics}")

        wandb.log({
            "training_time_hours": training_time,
            "final_eval_loss": final_metrics.get("eval_loss", 0),
            "final_eval_perplexity": final_metrics.get("eval_perplexity", 0)
        })

        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {model_dir}")
        logger.info(f"📊 Wandb run: {run_name}")

    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise
    finally:
        logger.info("🧹 Cleaning up resources...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        try:
            wandb.finish()
        except Exception:
            pass
        logger.info("✅ Cleanup complete")


if __name__ == "__main__":
    main()
