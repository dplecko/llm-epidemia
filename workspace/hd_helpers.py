import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier, XGBRegressor
import importlib.util
import os
from sklearn.model_selection import StratifiedKFold

def load_specs(dataset_name):
    """
    Load cond_spec and out_spec for a given dataset from the corresponding Python file.
    
    Args:
    - dataset_name (str): The name of the dataset (e.g., 'nsduh', 'nhanes').
    
    Returns:
    - tuple: (cond_spec, out_spec)
    """
    # Construct the path to the dataset file
    task_file = os.path.join("workspace", "tasks", f"tasks_{dataset_name}.py")
    
    # Dynamically import the task file
    spec = importlib.util.spec_from_file_location("task_module", task_file)
    task_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_module)
    
    # Extract cond_spec and out_spec
    cond_spec = getattr(task_module, f"{dataset_name}_cond", {})
    out_spec = getattr(task_module, f"{dataset_name}_out", {})
    
    return cond_spec, out_spec

def promptify(out_var, cond_vars, cond_row, dataset_name):
    # Load specs from the dataset file
    cond_spec, out_spec = load_specs(dataset_name)
    
    # Build the prompt
    prompt = "For a person"
    cond_parts = [cond_spec[var].format(cond_row[var]) for var in cond_vars]
    prompt += " " + ", ".join(cond_parts)
    prompt += ", " + out_spec[out_var]
    
    return prompt

def fit_lgbm(data, out_var, cond_vars, wgh_col=None, n_splits=5, seed=42):
    """
    Fit a LightGBM model to predict `out_var` based on `cond_vars` with out-of-bag predictions.
    """
    # Extract features and labels
    X = data[cond_vars].copy()
    y = data[out_var].astype("category").cat.codes
    weights = data[wgh_col].values if wgh_col else np.ones(len(y))

    # Convert categorical columns to category dtype
    for col in cond_vars:
        X[col] = X[col].astype("category")

    # Use fold indices to ensure reproducibility
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))

    # Create LightGBM dataset
    lgb_data = lgb.Dataset(X, label=y, weight=weights, categorical_feature=list(range(len(cond_vars))))
    
    # LightGBM parameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": seed,
    }

    # Cross-validation with manual fold tracking
    cv_results = lgb.cv(
        params,
        lgb_data,
        folds=folds,
        return_cvbooster=True,
        callbacks=[lgb.early_stopping(10)]
    )

    # Extract OOB predictions
    oob_preds = np.zeros(len(y))
    cvbooster = cv_results['cvbooster']
    
    # Use the manually provided folds for consistent test indices
    for booster, (train_idx, test_idx) in zip(cvbooster.boosters, folds):
        oob_preds[test_idx] = booster.predict(X.iloc[test_idx], num_iteration=booster.best_iteration)
    
    return oob_preds


def bootstrap_lgbm(data, out_var, cond_vars, wgh_col=None, n_splits=5, n_bootstraps=100, seed=42):
    np.random.seed(seed)
    y = data[out_var].astype("category").cat.codes
    weights = data[wgh_col].values if wgh_col else np.ones(len(y))
    n_samples = len(data)

    # Prepare storage for OOB predictions
    oob_preds_matrix = np.zeros((n_samples, n_bootstraps))

    for b in range(n_bootstraps):
        # Create bootstrap sample
        idx = np.arange(n_samples)
        boot_idx = np.random.choice(idx, size=n_samples, replace=True)

        # Create folds
        n_folds = n_splits
        folds = np.tile(np.arange(n_folds), int(np.ceil(len(boot_idx) / n_folds)))[:len(boot_idx)]
        np.random.shuffle(folds)

        # Prepare LightGBM dataset
        X_boot = data.iloc[boot_idx][cond_vars].copy()
        y_boot = y.iloc[boot_idx]
        weights_boot = weights[boot_idx]
        lgb_data = lgb.Dataset(X_boot, label=y_boot, weight=weights_boot, categorical_feature=list(range(len(cond_vars))))

        # LightGBM parameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": seed + b,
        }

        # Train model with manual folds
        cv_results = lgb.cv(
            params,
            lgb_data,
            folds=[(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(n_folds)],
            return_cvbooster=True,
            callbacks=[lgb.early_stopping(10)]
        )

        # Collect OOB predictions in the order of the original dataset
        oob_preds = np.zeros(n_samples)
        seen_idx = np.unique(boot_idx)  # All indices used in training

        # Populate OOB for seen points
        for f, booster in enumerate(cv_results['cvbooster'].boosters):
            test_idx = boot_idx[folds == f]
            oob_preds[test_idx] = booster.predict(data.iloc[test_idx][cond_vars], num_iteration=booster.best_iteration)

        # Predict for unseen points (using the first booster as a default)
        unseen_idx = np.setdiff1d(idx, seen_idx, assume_unique=True)
        if len(unseen_idx) > 0:
            oob_preds[unseen_idx] = cv_results['cvbooster'].boosters[0].predict(data.iloc[unseen_idx][cond_vars], num_iteration=cv_results['cvbooster'].boosters[0].best_iteration)

        # Store the predictions for this bootstrap run
        oob_preds_matrix[:, b] = oob_preds

    return oob_preds_matrix



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
