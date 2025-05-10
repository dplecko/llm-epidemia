
import re
import torch
import pdb
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
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


def xgb_conditional_prob(
    df: pd.DataFrame,
    target: str = "v1",
    test_size: float = 0.2,
    random_state: int = 42,
    model_kwargs: dict | None = None,
    return_model: bool = False,
) -> tuple[pd.DataFrame, Pipeline] | pd.DataFrame:
    """
    Fit XGBoost on df to estimate  P(target | other columns).
    Returns a dataframe of probabilities for every *observed*
    (v2, …, vn) combination, plus the fitted Pipeline if requested.

    Parameters
    ----------
    df : DataFrame containing v1, v2, …, vn
    target : column to predict (default 'v1')
    test_size, random_state : usual train/valid split args
    model_kwargs : overrides for XGBClassifier
    return_model : if True, also return the fitted pipeline
    """
    model_kwargs = model_kwargs or {}
    feats = [c for c in df.columns if c != target]

    # Identify categorical vs numeric feature columns
    cat = [c for c in feats if df[c].dtype == "object"
           or pd.api.types.is_categorical_dtype(df[c])]
    num = [c for c in feats if c not in cat]

    # Pre‑processor: one‑hot for categoricals, passthrough numerics
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ])

    # Decide classification vs regression from dtype / cardinality
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20:
        # treat as regression
        model = XGBRegressor(**model_kwargs)
    else:
        # classification
        n_classes = df[target].nunique()
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        model = XGBClassifier(objective=obj,
                              eval_metric="logloss",
                              use_label_encoder=False,
                              **model_kwargs)

    pipe = Pipeline([("prep", preprocess), ("model", model)])

    X_train, X_val, y_train, y_val = train_test_split(
        df[feats], df[target],
        test_size=test_size,
        stratify=df[target] if df[target].nunique() > 1 else None,
        random_state=random_state)

    pipe.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 1️⃣  Enumerate every UNIQUE combination of v2…vn present in data
    # ------------------------------------------------------------------
    combos = df[feats].drop_duplicates().reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2️⃣  Predict P(v1 | v2…vn)
    # ------------------------------------------------------------------
    if isinstance(model, XGBRegressor):
        combos[f"E[{target} | cond]"] = pipe.predict(combos)
    else:
        probs = pipe.predict_proba(combos)
        classes = pipe.named_steps["model"].classes_
        for i, cls in enumerate(classes):
            combos[f"P({target}={cls} | cond)"] = probs[:, i]

    return (combos, pipe) if return_model else combos



### Generalized extraction function ###
def extract_pv(prompt, levels, model_name, model, n_mc=128):
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
   
    return model.predict(prompt, levels, n_mc, max_batch_size,)



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