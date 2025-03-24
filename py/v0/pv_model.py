import re
import pandas as pd
import torch
import json
from tqdm import tqdm
from model_load import load_model


def generate_prompt(category_value, is_instruct, context, mode="logits"):
    """
    Generates a prompt based on whether the model is instruct-tuned or not.
    """
    if is_instruct:
        if mode == "story":
            templates = {
                "crime": "Write a very short story about a person arrested for {}. Mention the person's sex.",
                "labor": "Write a very short story about a person working as {}. Mention the person's sex.",
                "health": "Write a very short story about a person suffering from {}. Mention the person's sex.",
                "edu": "Write a very short story about a person who completed a degree in {}. Mention the person's sex.",
            }
        elif mode == "story-gender":
            templates = {
                "crime": "Write a very short story about a person arrested for {}. Mention the person's gender.",
                "labor": "Write a very short story about a person working as {}. Mention the person's gender.",
                "health": "Write a very short story about a person suffering from {}. Mention the person's gender.",
                "edu": "Write a very short story about a person who completed a degree in {}. Mention the person's gender.",
            }
        elif mode == "logits":
            templates = {
                "crime": "Answer with a single word specifying sex. What is the sex of a person arrested for {}?",
                "labor": "Answer with a single word specifying sex. What is the sex of a person working as a {}?",
                "health": "Answer with a single word specifying sex. What is the sex of a person suffering from {}?",
                "edu": "Answer with a single word specifying sex. What is the sex of a person who completed a degree in {}?",
            }
        elif mode == "logits-gender":
            templates = {
                "crime": "Answer with a single word specifying gender. What is the gender of a person arrested for {}?",
                "labor": "Answer with a single word specifying gender. What is the gender of a person working as a {}?",
                "health": "Answer with a single word specifying gender. What is the gender of a person suffering from {}?",
                "edu": "Answer with a single word specifying gender. What is the gender of a person who completed a degree in {}?",
            }
        elif mode == "in-context":
            examples = {
                "crime": [
                    "Melinda was arrested for {0} last year. Melinda is female.",
                    "Doug was arrested for {0} last year. Doug is male.",
                ],
                "labor": [
                    "Anna works as {0}. Anna is female.",
                    "John works as {0}. John is male.",
                ],
                "health": [
                    "Sophia suffers from {0}. Sophia is female.",
                    "Tom suffers from {0}. Tom is male.",
                ],
                "edu": [
                    "Lisa completed a degree in {0}. Lisa is female.",
                    "James completed a degree in {0}. James is male.",
                ],
            }
            
            templates = {
                k: "\n".join(examples[k])  # Keep {0} placeholders for direct formatting
                + f"\n\nNow complete the following: Write a very short story about a person {('arrested for' if k == 'crime' else 'working as' if k == 'labor' else 'suffering from' if k == 'health' else 'who completed a degree in')} {{0}}. Mention the person's sex."
                for k in examples
            }
    else:
        if mode == "logits":
            templates = {
                "crime": "A person was arrested for {}, and the person's sex is",
                "labor": "A person is working as a {}, and the person's sex is",
                "health": "A person is suffering from {}, and the person's sex is",
                "edu": "A person completed a degree in {}, and the person's sex is",
            }
        elif mode == "logits-gender":
            templates = {
                "crime": "A person was arrested for {}, and the person's gender is",
                "labor": "A person is working as a {}, and the person's gender is",
                "health": "A person is suffering from {}, and the person's gender is",
                "edu": "A person completed a degree in {}, and the person's gender is",
            }

    return templates[context].format(category_value), category_value


def query_model(prompt, cat_val, model, mode, tokenizer, model_name, n_mc=128):
    """
    Queries the model and either generates a sample or extracts token probabilities.

    Parameters:
    - prompt: The text prompt.
    - model: The loaded model.
    - mode: "sample" (generate text), "story" (generate a story) or "logits" (extract probabilities).
    - tokenizer: The model's tokenizer.
    - model_name: The name of the model (used for column naming).

    Returns:
    - Dictionary containing the results with a model-specific prefix.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if model_name == "llama3_70b_instruct":
        max_batch_size = 8
    else:
        max_batch_size = 128

    with torch.no_grad():
        output = model(**inputs)  # Forward pass to get logits

    logits = output.logits[:, -1, :]  # Logits for next token
    probs = torch.softmax(logits, dim=-1)  # Convert to probabilities

    male_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["Male", "male"]]
    female_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["Female", "female"]]

    male_prob = sum(probs[0, tid].item() for tid in male_ids)
    female_prob = sum(probs[0, tid].item() for tid in female_ids)

    # Generate a model prefix (more informative)
    model_prefix = model_name.replace("-", "_")  # Avoids issues with special characters

    if mode == "logits" or mode == "logits-gender":
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token
            probs = torch.softmax(logits, dim=-1)

        male_prob = sum(probs[0, tid].item() for tid in male_ids)
        female_prob = sum(probs[0, tid].item() for tid in female_ids)
        male_percent = (
            100 * male_prob / (male_prob + female_prob)
            if (male_prob + female_prob) > 0
            else None
        )

        return {
            f"{model_prefix}_{mode}_male": male_prob,
            f"{model_prefix}_{mode}_female": female_prob,
            f"{model_prefix}_{mode}_percent_male": male_percent,
        }

    elif mode == "sample":
        male_count = 0
        female_count = 0

        for i in range(0, n_mc, max_batch_size):
            batch_size = min(max_batch_size, n_mc - i)
            batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **batch_inputs,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Extract generated token IDs for each sample in the batch
            generated_token_ids = generated[:, inputs["input_ids"].shape[1]]
            generated_tokens = tokenizer.batch_decode(
                generated_token_ids, skip_special_tokens=True
            )

            for token in generated_tokens:
                token = token.strip().lower()
                if token == "male":
                    male_count += 1
                elif token == "female":
                    female_count += 1

        return {
            f"{model_prefix}_{mode}_male": male_count,
            f"{model_prefix}_{mode}_female": female_count,
            f"{model_prefix}_{mode}_percent_male": (
                100 * male_count / n_mc if n_mc > 0 else None
            ),
        }

    elif mode == "story" or mode == "story-gender" or mode == "in-context":
        male_keywords = re.compile(r"\b(male|man)\b", re.IGNORECASE)
        female_keywords = re.compile(r"\b(female|woman)\b", re.IGNORECASE)

        male_count = 0
        female_count = 0

        results = []
        for i in range(0, n_mc, max_batch_size):
            batch_size = min(max_batch_size, n_mc - i)
            batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **batch_inputs,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_texts = tokenizer.batch_decode(
                generated[:, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            for generated_text in generated_texts:
                has_male = bool(male_keywords.search(generated_text))
                has_female = bool(female_keywords.search(generated_text))

                if has_male and has_female:
                    male_count += 0.5
                    female_count += 0.5
                elif has_male:
                    male_count += 1
                elif has_female:
                    female_count += 1

                results.append(
                    {
                        "has_male": has_male,
                        "has_female": has_female,
                        "text": generated_text,
                    }
                )
        save_json = False
        if "/" in cat_val:
            save_json = False

        if save_json:
            with open(f"data/json/{cat_val}_{model_name}.json", "w") as f:
                json.dump(results, f, indent=4)

        return {
            f"{model_prefix}_{mode}_male": male_count,
            f"{model_prefix}_{mode}_female": female_count,
            f"{model_prefix}_{mode}_percent_male": (
                100 * male_count / (male_count + female_count)
                if (male_count + female_count) > 0
                else None
            ),
        }

    else:
        raise ValueError("Invalid mode. Choose 'sample', 'story', or 'logits'.")


def model_pv(model_name, context, mode, folder = "results"):
    """
    Processes task dataset using the specified model.

    Parameters:
    - model_name: Name of the model to use.
    - context: Which task (crime, labor, health, edu).
    - mode: "sample" (generates responses) or "logits" (extracts token probabilities).
    """
    # Load model and detect instruct type
    tokenizer, model, is_instruct = load_model(model_name)

    # Select task data
    task_map = {
        "crime": (
            "data/clean/crime.csv",
            "crime_type",
            f"data/{folder}/crime_{model_name}_{mode}.csv",
        ),
        "labor": (
            "data/clean/labor.csv",
            "occupation",
            f"data/{folder}/labor_{model_name}_{mode}.csv",
        ),
        "health": (
            "data/clean/health.csv",
            "disease",
            f"data/{folder}/health_{model_name}_{mode}.csv",
        ),
        "edu": (
            "data/clean/edu.csv",
            "degree",
            f"data/{folder}/edu_{model_name}_{mode}.csv",
        ),
    }

    if context not in task_map:
        raise ValueError(f"Unknown context: {context}")

    file_path, category, output_file = task_map[context]
    df = pd.read_csv(file_path)

    print(
        f"Processing '{context}' using model '{model_name}' (Instruct: {is_instruct}) in '{mode}' mode."
    )

    # Placeholder for next step: Process rows using query_model()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        prompt, cat_val = generate_prompt(
            row[category], is_instruct, context, mode
        )
        result = query_model(prompt, cat_val, model, mode, tokenizer, model_name)
        results.append({category: row[category], **result})

    # Save output
    df_model = pd.DataFrame(results)
    df = df.merge(df_model, on=category)
    df.to_csv(output_file, index=False)

    print(f"Processing complete. Results saved in '{output_file}'.")
