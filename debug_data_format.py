#!/usr/bin/env python3
"""
Debug script to understand the data format issue
"""

import json
from transformers import AutoTokenizer
from datasets import Dataset

def debug_data_format():
    """Debug the data format issue."""
    print("🔍 Debugging data format...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load a small sample of data
    with open("finetuning_setup/data/llama_formatted_data.json", 'r') as f:
        data = json.load(f)
    
    # Take first 3 examples
    texts = [item['text'] for item in data[:3]]
    dataset = Dataset.from_dict({"text": texts})
    
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"📊 First example text length: {len(texts[0])}")
    print(f"📊 First example preview: {texts[0][:200]}...")
    
    # Test tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None  # Return lists, not tensors
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    print(f"📊 Tokenized dataset size: {len(tokenized_dataset)}")
    print(f"📊 Tokenized dataset features: {tokenized_dataset.features}")
    
    # Check the first example
    first_example = tokenized_dataset[0]
    print(f"📊 First tokenized example keys: {first_example.keys()}")
    print(f"📊 Input IDs type: {type(first_example['input_ids'])}")
    print(f"📊 Input IDs length: {len(first_example['input_ids'])}")
    print(f"📊 Input IDs first 10: {first_example['input_ids'][:10]}")
    
    # Check if input_ids is nested
    if isinstance(first_example['input_ids'][0], list):
        print("❌ PROBLEM: input_ids is nested (list of lists)")
        print(f"   First element type: {type(first_example['input_ids'][0])}")
        print(f"   First element length: {len(first_example['input_ids'][0])}")
    else:
        print("✅ input_ids is flat (list of ints)")
    
    # Test with a single example
    print("\n🔧 Testing single example tokenization...")
    single_text = texts[0]
    single_tokenized = tokenizer(
        single_text,
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors=None
    )
    
    print(f"📊 Single tokenized keys: {single_tokenized.keys()}")
    print(f"📊 Single input_ids type: {type(single_tokenized['input_ids'])}")
    print(f"📊 Single input_ids length: {len(single_tokenized['input_ids'])}")
    
    # Test with return_tensors="pt"
    print("\n🔧 Testing with return_tensors='pt'...")
    single_tokenized_pt = tokenizer(
        single_text,
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors="pt"
    )
    
    print(f"📊 PT tokenized keys: {single_tokenized_pt.keys()}")
    print(f"📊 PT input_ids shape: {single_tokenized_pt['input_ids'].shape}")
    print(f"📊 PT input_ids type: {type(single_tokenized_pt['input_ids'])}")

if __name__ == "__main__":
    debug_data_format()
