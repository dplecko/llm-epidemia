
import re
from collections import defaultdict
import numpy as np

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

def determine_possible_levels(possible_levels, tmp_probs):
    """
    Determine possible levels, that is, likelihood probabilities -> makes sure the sum of probabilities is not larger than 100%.

    Returns:
        
    """
    selected_likelihood_index = np.argmax(tmp_probs)
    selected_likelihood = possible_levels[selected_likelihood_index]

    index = len(possible_levels) - selected_likelihood_index
    return possible_levels[:index]
    

def extract_pv(prompt, levels, model_name, model, task_spec, pos_ans=None, n_mc=128):
    """
    Unified interface for extracting predicted values using sampling or logits.

    Args:
        prompt (str): Initial text prompt.
        levels (List[List[str]] or None): Optional levels for classification.
        model_name (str): Model name, used to set batch size.
        model: Language model used for generation or logits.
        n_mc (int): Number of samples for sampling modes.

    Returns:
        List[float, int, or None]: Extracted values based on mode and config.
    """
    
    if model_name == "llama3_70b_instruct":
        max_batch_size = 32
    else:
        max_batch_size = 128
        
    if isinstance(prompt, list):
        prob_matrix = np.zeros((len(prompt), len(levels)))
        level_matrix = []
        model_texts = []
        possible_levels = levels
        for i, p in enumerate(prompt):
            if len(possible_levels) > 0:
                tmp_levels, tmp_probs, tmp_texts = model.predict(p, possible_levels, n_mc, max_batch_size,)
                prob_matrix[i, :len(tmp_probs)] = tmp_probs
                level_matrix.append(tmp_levels)
                model_texts.append(tmp_texts)
                possible_levels = determine_possible_levels(possible_levels, tmp_probs)
        return prob_matrix, level_matrix, model_texts
    else:
        return model.predict(prompt, levels, n_mc, max_batch_size,)

def extract_pv_batch(prompts, levels, model_name, model, task_spec, n_mc=128, prob=False):
    max_batch = 32 if model_name == "llama3_70b_instruct" else 128
    return model.predict_batch(  # type: ignore[attr‑defined]
        prompts=prompts,
        levels=levels,
        num_permutations=n_mc,
        max_batch_size=max_batch,
        prob=prob
    )
