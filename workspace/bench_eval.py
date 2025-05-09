
import numpy as np
import pandas as pd
import json
import pdb

import sys
import os
sys.path.append(os.path.abspath("workspace"))
from metrics import ks_w, cat_to_distr

def eval_cts(res, model_name, mode, dataset, v1, v2):
    rows = []

    for r in res:
        n_mc = len(r["model_vals"])
        val_true = np.array(r["true_vals"], dtype=float)
        wgh_true = np.array(r.get("weights", [1.0] * len(val_true)))
        val_mod = np.array([m for m in r["model_vals"] if m is not None])

        # Bootstrap best error
        best_err = []
        for _ in range(100):
            idx = np.random.choice(len(val_true), size=n_mc, p=wgh_true / wgh_true.sum(), replace=True)
            val_true_bt = val_true[idx]
            test_bt = ks_w(val_true_bt, val_true, w_x=None, w_y=wgh_true)
            best_err.append(test_bt)
        best_err = np.quantile(best_err, 0.975)

        # Worst-case (random uniform)
        worst_bt = np.random.uniform(val_true.min(), val_true.max(), size=1000)
        worst_err = ks_w(val_true, worst_bt, w_x=wgh_true, w_y=None)

        # Actual model error
        score = ks_w(val_true, val_mod, w_x=wgh_true, w_y=None)
        bench = (score - worst_err) / (best_err - worst_err)
        bench = 100 * max(min(bench, 1), 0)

        rows.append({
            "p_true": float(np.average(val_true, weights=wgh_true)),
            "p_mod": float(np.mean(val_mod)),
            "bench": bench,
            "cond": r["condition"],
            "true_vals": {"vals": val_true.tolist(), "wgh": wgh_true.tolist()},
            "mod_vals": {"vals": val_mod.tolist()},
        })

    return pd.DataFrame(rows)

def eval_bin(res, model_name, mode, dataset, v1, v2):
    rows = []

    for r in res:
        n_mc = len(r["model_vals"])
        val_true = np.array(r["true_vals"], dtype=float)
        wgh_true = np.array(r.get("weights", [1.0] * len(val_true)))
        p_true = float(np.average(val_true, weights=wgh_true))

        if n_mc == 2:
            p_mod = r["model_vals"][1]
        else:
            val_mod = np.array([m for m in r["model_vals"] if m is not None])
            if val_mod is None or len(val_mod) == 0:
                print("No good answers found for model:", model_name, "v1:", v1, "v2:", v2, "condition:", r["condition"])
                p_mod = 0.5
                n_mc = 128
            else:
                p_mod = np.mean(val_mod)
                n_mc = len(val_mod)
        
        if n_mc == 2:
            n_mc = 10**6  # inflate for stability

        best_err = 2 * np.sqrt(p_true * (1 - p_true) / n_mc)
        worst_err = 0.5 * (p_true**2 + (1 - p_true)**2)
        score = abs(p_true - p_mod)

        bench = (score - worst_err) / (best_err - worst_err)
        bench = 100 * max(min(bench, 1.0), 0.0)

        rows.append({
            "p_true": p_true,
            "p_mod": p_mod,
            "bench": bench,
            "cond": r["condition"]
        })

    return pd.DataFrame(rows)

def eval_cat(res, model_name, mode, dataset, v1, v2, levels):
    nbins = len(levels)
    lvl_names = levels
    rows = []
    distr_rows = []

    for r in res:
        n_mc = len(r["model_vals"])
        pdb.set_trace()
        val_true = np.array(r["true_vals"], dtype=int)
        wgh_true = np.array(r.get("weights", [1.0] * len(val_true)))
        mod_vals = np.array([m for m in r["model_vals"] if m is not None])

        distr_true = cat_to_distr(val_true, wgh_true, nbins)
        distr_mod = cat_to_distr(mod_vals, None, nbins)

        for i in range(nbins):
            distr_rows.append({
                "lvl": i + 1,
                "lvl_names": lvl_names[i],
                "prop": distr_true[i],
                "type": "Reality",
                "cond": r["condition"]
            })
            distr_rows.append({
                "lvl": i + 1,
                "lvl_names": lvl_names[i],
                "prop": distr_mod[i],
                "type": "Model",
                "cond": r["condition"]
            })

        # Bootstrap best error
        best_err = []
        for _ in range(100):
            idx = np.random.choice(len(val_true), size=n_mc, p=wgh_true / wgh_true.sum(), replace=True)
            val_bt = val_true[idx]
            distr_bt = cat_to_distr(val_bt, None, nbins)
            best_err.append(np.abs(distr_bt - distr_true).sum())
        best_err = np.quantile(best_err, 0.975)

        worst_bt = np.ones(nbins) / nbins
        worst_err = np.abs(worst_bt - distr_true).sum()
        score = np.abs(distr_true - distr_mod).sum()
        
        # only scores are essentially 0 and 100!
        if worst_err < best_err :
            worst_err = best_err + 0.0001

        bench = (score - worst_err) / (best_err - worst_err)
        bench = 100 * max(min(bench, 1), 0)

        rows.append({
            "bench": bench,
            "cond": r["condition"]
        })

    df = pd.DataFrame(rows)
    df.attrs["distr"] = pd.DataFrame(distr_rows)
    return df

def eval_to_score(df):
    return df["bench"].mean()

import re
def dat_name_clean(path):
    base = path.split("/")[-1]
    return re.sub(r"\.parquet$", "", base)

def eval_task(model_name, task):

    mode = "logits"
    dataset = dat_name_clean(task["dataset"])
    
    v1 = task["variables"][0]
    v2 = task["variables"][1] if len(task["variables"]) > 1 else None
    levels = pd.read_parquet(task["dataset"])[v1].unique().tolist()

    parts = [model_name, dataset, v1]
    if v2: parts.append(v2)
    fname = "_".join(parts) + ".json"
    path = os.path.join("data", "benchmark", fname)

    with open(path, "r") as f:
        res = json.load(f)

    if levels is not None and len(levels) == 2:
        return eval_bin(res, model_name, mode, dataset, v1, v2)
    else:
        return eval_cat(res, model_name, mode, dataset, v1, v2, levels)
    # else:
    #     return eval_cts(res, model_name, mode, dataset, v1, v2)