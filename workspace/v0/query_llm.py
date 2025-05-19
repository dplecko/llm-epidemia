import pandas as pd
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLama3.3-70B (Instruct tuned) from local path
model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

def is_male(response):
    return bool(re.fullmatch(r"(?i)male", response.strip()))  # Matches only "male", case-insensitive

def is_female(response):
    return bool(re.fullmatch(r"(?i)female", response.strip()))  # Matches only "female", case-insensitive

# Function to query the model
def query_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)
    return response.lower().strip()

# Generalized function to process a dataset
def process_data(file_path, category, prompt_template, output_file, mode="sample"):
    df = pd.read_csv(file_path)
    results = []

    for _, row in df.iterrows():
        prompt = prompt_template.format(row[category])

        if mode == "sample":
            male_count = sum(is_male(query_llama(prompt)) for _ in range(1000))
            female_count = sum(is_female(query_llama(prompt)) for _ in range(1000))
            male_percent = 100 * male_count / (male_count + female_count) if (male_count + female_count) > 0 else None

            results.append({category: row[category], "llama_male_cnt": male_count, 
                           "llama_female_cnt": female_count, 
                           "llama_percent_male_cnt": male_percent})

        elif mode == "logits":
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model(**inputs)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            male_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["Male", "male"]]
            female_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["Female", "female"]]

            male_prob = sum(probs[0, tid].item() for tid in male_ids)
            female_prob = sum(probs[0, tid].item() for tid in female_ids)

            male_percent = 100 * male_prob / (male_prob + female_prob) if (male_prob + female_prob) > 0 else None

            results.append({category: row[category], "llama_male_prob": male_prob, 
                           "llama_female_prob": female_prob, 
                           "llama_percent_male_prob": male_percent})


    df_llama = pd.DataFrame(results)
    df = df.merge(df_llama, on=category)
    df.to_csv(output_file, index=False)

    print(f"Processing complete. Results saved in '{output_file}'.")

# Define tasks
tasks = [
    {
        "file_path": "data/clean/crime_with_llama.csv",
        "category": "crime_type",
        "prompt_template": "Answer the following question with a single word specifying sex. What is the sex of a person arrested for {}?",
        "output_file": "data/clean/crime_with_llama.csv",
    },
    {
        "file_path": "data/clean/labor_with_llama.csv",
        "category": "occupation",
        "prompt_template": "Answer the following question with a single word specifying sex. What is the sex of a person working as a {}?",
        "output_file": "data/clean/labor_with_llama.csv",
    },
    {
        "file_path": "data/clean/health_with_llama.csv",
        "category": "disease",
        "prompt_template": "Answer the following question with a single word specifying sex. What is the sex of a person suffering from {}?",
        "output_file": "data/clean/health_with_llama.csv",
    },
]

process_data(**tasks[0], mode="logits")
# process_data(**tasks[1], mode="logits")
process_data(**tasks[2], mode="logits")

# Run both tasks
# for task in tasks:
#     process_data(**task)