#!/usr/bin/env python3
"""
Test script for the fixed data collator
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset

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

def test_collator():
    """Test the fixed data collator."""
    print("🔍 Testing fixed data collator...")
    
    # Setup directories
    base_dir = Path("/local/eb/shreyas")
    cache_dir = base_dir / "huggingface_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        cache_dir=str(cache_dir)
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load a small sample of data
    with open("finetuning_setup/data/llama_formatted_data.json", 'r') as f:
        data = json.load(f)
    
    # Use only first 5 examples for testing
    texts = [item['text'] for item in data[:5]]
    dataset = Dataset.from_dict({"text": texts})
    
    print(f"📊 Dataset size: {len(dataset)}")
    
    # Tokenize dataset
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
    print(f"📊 Tokenized dataset size: {len(tokenized_dataset)}")
    print(f"📊 Tokenized dataset features: {tokenized_dataset.features}")
    
    # Test the data collator
    data_collator = FixedDataCollator(tokenizer=tokenizer, mlm=False)
    
    # Test with a small batch
    sample_batch = tokenized_dataset.select(range(3))
    print(f"📊 Sample batch size: {len(sample_batch)}")
    
    # Convert to list of features
    features = [sample_batch[i] for i in range(len(sample_batch))]
    print(f"📊 Features type: {type(features)}")
    print(f"📊 First feature keys: {features[0].keys()}")
    print(f"📊 First feature input_ids length: {len(features[0]['input_ids'])}")
    print(f"📊 First feature attention_mask length: {len(features[0]['attention_mask'])}")
    
    try:
        batch = data_collator(features)
        print("✅ Data collator test passed!")
        print(f"📊 Batch keys: {batch.keys()}")
        print(f"📊 Input IDs shape: {batch['input_ids'].shape}")
        print(f"📊 Attention mask shape: {batch['attention_mask'].shape}")
        print(f"📊 Labels shape: {batch['labels'].shape}")
        print(f"📊 Input IDs dtype: {batch['input_ids'].dtype}")
        print(f"📊 Attention mask dtype: {batch['attention_mask'].dtype}")
        print(f"📊 Labels dtype: {batch['labels'].dtype}")
        
        # Test that all sequences are the same length
        input_lengths = [len(seq) for seq in batch['input_ids']]
        print(f"📊 Input lengths: {input_lengths}")
        print(f"📊 All lengths equal: {len(set(input_lengths)) == 1}")
        
    except Exception as e:
        print(f"❌ Data collator test failed: {e}")
        raise

if __name__ == "__main__":
    test_collator()
