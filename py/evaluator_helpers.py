
import re
import torch
import pdb
from collections import defaultdict

def max_lvl_len(levels, tokenizer):
    """
    Compute the maximum tokenized word length across all levels.

    Args:
        levels (List[List[str]]): A list of levels, where each level is a list of words.
        tokenizer: A tokenizer with a .tokenize method (e.g., HuggingFace tokenizer).

    Returns:
        int: Maximum number of tokens for any single word across all levels.
    """
    return max(max(len(tokenizer.tokenize(word)) for word in lvl) for lvl in levels)

def lvl_probs(model, tokenizer, inputs, levels):
    
    """
    Compute normalized probabilities for each factor level based on model's logits.

    Args:
        model: A language model that outputs logits.
        tokenizer: Tokenizer with convert_tokens_to_ids method.
        inputs (dict): Tokenized input tensors for the model.
        levels (List[List[str]]): List of levels, each a list of word strings.

    Returns:
        List[float] or None: Normalized probabilities over levels or None if levels is None.
    """
    if levels is None:
        return None

    # Convert words in levels to token IDs
    level_ids = [[tokenizer.convert_tokens_to_ids(t) for t in lvl] for lvl in levels]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        probs = torch.softmax(logits, dim=-1)

    level_probs = [sum(probs[0, tid].item() for tid in lvl_ids) for lvl_ids in level_ids]
    total_prob = sum(level_probs)
    return [p / total_prob for p in level_probs], None

def txt_to_lvl(text, levels):
    """
    Match input text to a level using regex. Handles ambiguity.

    Args:
        text (str): The generated text to analyze.
        levels (List[List[str]]): List of levels, each a list of words.

    Returns:
        int, float, or None: Index of matched level, 0.5 if ambiguous, or None if no match.
    """
    if levels is None:
        return None
    
    level_patterns = [
        re.compile(r"\b(" + "|".join(set(map(str.lower, lvl))) + r")\b", re.IGNORECASE)
        for lvl in levels
    ]
    
    level_matches = [bool(pattern.search(text)) for pattern in level_patterns]

    if sum(level_matches) > 1:
        return 0.5  # Ambiguous match
    elif any(level_matches):
        return level_matches.index(True)  # Return the index of the matched level
    else:
        return None  # No match found

def txt_to_num(text):
    """
    Extract the first numeric value from text.

    Args:
        text (str): Text containing potential numeric values.

    Returns:
        float or None: First numeric value found, or None if no match.
    """
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    return float(matches[0]) if matches else None

def lvl_sample(model, tokenizer, inputs, levels, n_mc, max_batch_size):
    """
    Generate multiple completions and classify each to a level.

    Args:
        model: Language model for generation.
        tokenizer: Tokenizer with .generate and .batch_decode methods.
        inputs (dict): Tokenized input tensors.
        levels (List[List[str]]): List of levels for classification.
        n_mc (int): Total number of samples to generate.
        max_batch_size (int): Max samples per batch.

    Returns:
        List[Union[int, float, None]]: Level index or 0.5/None per sample.
    """
    samples = []
    for i in range(0, n_mc, max_batch_size):
        batch_size = min(max_batch_size, n_mc - i)
        batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **batch_inputs,
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        # Extract generated tokens
        generated_tokens = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Map text to levels
        samples.extend(txt_to_lvl(token, levels) for token in generated_tokens)

    return samples, generated_tokens

def cts_sample(model, tokenizer, inputs, n_mc, max_batch_size, max_tokens=10):
    """
    Generate numeric answers from text completions.

    Args:
        model: Language model for generation.
        tokenizer: Tokenizer used to decode outputs.
        inputs (dict): Tokenized input tensors.
        n_mc (int): Number of completions to sample.
        max_batch_size (int): Max samples per generation batch.
        max_tokens (int): Max tokens to generate per sample.

    Returns:
        List[float or None]: Extracted numeric values from completions.
    """
    samples = []
    for i in range(0, n_mc, max_batch_size):
        batch_size = min(max_batch_size, n_mc - i)
        batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **batch_inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        # Extract generated text
        generated_texts = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Convert text to numeric values
        samples.extend(txt_to_num(text) for text in generated_texts)

    return samples, generated_texts

def story_sample(model, tokenizer, inputs, second_prompt, levels, n_mc, max_batch_size, max_tokens=50):
    """
    Generate a story, follow up with a question, and extract a level or numeric answer.

    Args:
        model: Language model for generation.
        tokenizer: Tokenizer for prompt construction and decoding.
        inputs (dict): Initial prompt inputs for story generation.
        second_prompt (str): Follow-up question to ask about the story.
        levels (List[List[str]] or None): Levels for classification (optional).
        n_mc (int): Number of samples to generate.
        max_batch_size (int): Max samples per generation batch.
        max_tokens (int): Max tokens for story generation.

    Returns:
        List[Union[int, float, None]]: Extracted level index or numeric value per sample.
    """
    samples = []
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    
    for i in range(0, n_mc, max_batch_size):
        batch_size = min(max_batch_size, n_mc - i)
        batch_inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

        with torch.no_grad():
            generated_story = model.generate(
                **batch_inputs,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        generated_texts = tokenizer.batch_decode(
            generated_story[:, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Construct the follow-up prompt
        followup_prompts = [
            f"Based on the following text: {text} answer the following question: {second_prompt}"
            for text in generated_texts
        ]

        # Tokenize follow-up prompts
        followup_inputs = tokenizer(followup_prompts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            generated_responses = model.generate(
                **followup_inputs,
                max_new_tokens=3,  # Short answer expected
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract input length to remove from generated responses
        input_length = followup_inputs["input_ids"].shape[1]

        # Decode only the newly generated portion
        response_texts = tokenizer.batch_decode(generated_responses[:, input_length:], skip_special_tokens=True)

        for response in response_texts:
            if levels is not None:
                samples.append(txt_to_lvl(response, levels))
            else:
                samples.append(txt_to_num(response))

    return samples, generated_texts

### Generalized extraction function ###
def extract_pv(prompt, levels, mode, model_name, model, tokenizer, second_prompt=None, n_mc=128):
    """
    Unified interface for extracting predicted values using sampling or logits.

    Args:
        prompt (str): Initial text prompt.
        levels (List[List[str]] or None): Optional levels for classification.
        mode (str): One of {"logits", "sample", "story"}.
        model_name (str): Model name, used to set batch size.
        model: Language model used for generation or logits.
        tokenizer: Tokenizer to prepare input and decode output.
        second_prompt (str, optional): Follow-up question for "story" mode.
        n_mc (int): Number of samples for sampling modes.

    Returns:
        List[float, int, or None]: Extracted values based on mode and config.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if model_name == "llama3_70b_instruct":
        max_batch_size = 32
    else:
        max_batch_size = 128

    if levels is not None:
        max_tokens = max_lvl_len(levels, tokenizer)

    if mode == "logits" and max_tokens == 1:
        return lvl_probs(model, tokenizer, inputs, levels)

    elif (mode == "sample" and levels is not None) or (mode == "logits" and max_tokens > 1):
        return lvl_sample(model, tokenizer, inputs, levels, n_mc, max_batch_size)
    
    elif mode == "sample" and levels is None:
        return cts_sample(model, tokenizer, inputs, n_mc, max_batch_size)

    elif mode == "story":
        if second_prompt is None:
            raise ValueError("second_prompt is required for 'story' mode")
        return story_sample(model, tokenizer, inputs, second_prompt, levels, n_mc, max_batch_size)

    else:
        raise ValueError("Invalid mode. Choose 'sample', 'story', or 'logits'.")

def d2d_wgh_col(dataset):

    if "nhanes" in dataset:
        mc_wgh_col = "mec_wgh"
    elif "census" in dataset:
        mc_wgh_col = "weight"
    elif "gss" in dataset:
        mc_wgh_col = "wgh"
    else:
        mc_wgh_col = None

    return mc_wgh_col

def compress_vals(true_vals, wghs):
    agg = defaultdict(float)
    for v, w in zip(true_vals, wghs):
        agg[v] += w
    items = agg.items()
    vals, weights = zip(*items)
    return list(vals), list(weights)