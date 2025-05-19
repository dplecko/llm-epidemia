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

def load_specs(dataset_name, prob=False):
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
    
    if prob:
        out_spec = getattr(task_module, f"{dataset_name}_pout", {})
    
    return cond_spec, out_spec

def promptify(out_var, cond_vars, cond_row, dataset_name, prob=False):
    # Load specs from the dataset file
    cond_spec, out_spec = load_specs(dataset_name, prob=prob)
    
    # Build the prompt
    prompt = "For a person"
    cond_parts = [cond_spec[var].format(cond_row[var]) for var in cond_vars]
    prompt += " " + ", ".join(cond_parts)
    prompt += ", " + out_spec[out_var]
    
    return prompt


def generate_probability_levels():
    levels = ['0%']
    for i in range(20):
        level = f"{i * 5}% - {(i + 1) * 5}%"
        levels.append(level)
    levels.append('100%')
    return levels

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

    for col in cond_vars:
        if not isinstance(data[col], pd.CategoricalDtype):
            data[col] = data[col].astype("category")
    
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
