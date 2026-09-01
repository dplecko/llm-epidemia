"""Rescore BRFSS tasks against 2015/2018/2020/2023 ground truth, model output
held fixed: the 9 low-dimensional by-state tasks (reuses eval_cat /
compress_vals, JSON predictions) and the 11 income-free high-dimensional
tasks (reuses eval_hd / fit_lgbm, parquet predictions, ground truth refit per
year -- the 28 income-conditioned tasks are excluded, see NOTES). Run from
the repo root:
    python workspace/brfss_temporal_eval.py
Writes data/benchmark/brfss_temporal_scores.csv (columns include `setting`:
low/high).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace", "utils"))
sys.path.append(os.path.join(os.getcwd(), "workspace", "tasks"))
import eval as eval_mod
from eval import eval_cat, eval_hd               # repo's scoring formulas, unmodified
from extract_helpers import compress_vals         # repo's aggregation helper, unmodified
from hd_helpers import fit_lgbm                    # repo's ground-truth fitter, unmodified
from helpers import task_to_filename, dat_name_clean
from tasks_brfss import tasks_brfss, tasks_brfss_hd  # repo's task specs, unmodified

YEARS = [2015, 2018, 2020, 2023]
MODELS = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct",
          "phi4", "gemma3_27b_instruct", "deepseek_7b_chat"]
YEAR_PARQUET = {y: ("data/clean/brfss.parquet" if y == 2023 else f"data/clean/brfss_{y}.parquet")
                for y in YEARS}
OUT_CSV = "data/benchmark/brfss_temporal_scores.csv"

# income conditions on a scale that isn't comparable across years (5-cat
# _INCOMG pre-2021 vs. 7-cat _INCOMG1 in 2023) -- excluded from the hd set.
HD_TASKS = [t for t in tasks_brfss_hd if "income" not in t["v_cond"]]


def get_ground_truth(data, task_spec):
    # Verbatim copy of workspace/extract.py:14-15 (a pure 1-line function, no
    # model coupling). Not imported directly because extract.py's module-level
    # `from common import *` pulls in transformers/model-loading we don't need.
    return data[task_spec["variables"][0]].tolist()


def build_res_for_year(orig_res, year_df, task_spec):
    """Low-dim: swap true_vals/true_weights/n_data/total_weight to `year_df`'s
    ground truth per state condition; leave model_vals/model_weights/
    model_texts (the frozen model output) untouched."""
    out = []
    for r in orig_res:
        filtered = year_df[year_df["state"] == r["condition"]]
        if filtered.empty:
            continue  # territory didn't participate in BRFSS that year (e.g. Virgin Islands pre-2023)
        true_vals = get_ground_truth(filtered, task_spec)
        weights = filtered["weight"].tolist() if "weight" in filtered.columns else [1] * len(true_vals)
        true_vals, true_weights = compress_vals(true_vals, weights)
        out.append({**r, "true_vals": true_vals, "true_weights": true_weights,
                    "n_data": len(filtered), "total_weight": sum(true_weights)})
    return out


# eval_cat's cat_to_distr (workspace/utils/metrics.py) loops in pure Python per
# row inside its 100-draw best_err bootstrap. In production that runs once per
# task against 2023 and is cached forever; here it reruns cache-cold for every
# (task, year) pair, which times out. Swap in a numerically-identical
# vectorized version (np.bincount) for performance only -- the repo file on
# disk is untouched.
def _fast_cat_to_distr(x, w, nbins):
    x = np.asarray(x)
    w = np.ones_like(x, dtype=float) if w is None else np.asarray(w, dtype=float)
    distr = np.bincount(x, weights=w, minlength=nbins)[:nbins]
    return distr / distr.sum()


def score_lowdim(year_df):
    rows = []
    for task in tasks_brfss:
        v1, v2 = task["variables"][0], task["variables"][1]
        dataset = dat_name_clean(task["dataset"])
        levels = pd.read_parquet(task["dataset"])[v1].unique().tolist()

        for model in MODELS:
            with open(os.path.join("data/benchmark", task_to_filename(model, task))) as f:
                orig_res = json.load(f)

            for year in YEARS:
                df_y = year_df[year]
                available = df_y[v1].notna().any()
                score = None
                if available:
                    res = build_res_for_year(orig_res, df_y, task)
                    out = eval_cat(res, f"{dataset}_{year}", v1, v2, levels, cache_dir="data/benchmark")
                    score = out["bench"].iloc[0]
                rows.append({"setting": "low", "outcome": v1, "cond_vars": "state", "model": model,
                             "year": year, "score": score, "available": available})
                print(f"[low]  {v1:16s} {model:22s} {year}  {'—' if score is None else f'{score:6.2f}'}", flush=True)
    return rows


def score_hd(year_df):
    rows = []
    for task in HD_TASKS:
        v_out, cond_vars = task["v_out"], task["v_cond"]

        # Ground truth (lgbm_pred) depends only on (task, year), not on the
        # model -- fit it once per year here instead of once per (year, model).
        gt_by_year = {}
        for year in YEARS:
            sub = year_df[year][cond_vars + [v_out, "weight"]].dropna().copy()
            if sub.empty:
                gt_by_year[year] = None
                continue
            sub["lgbm_pred"] = fit_lgbm(sub, v_out, cond_vars, wgh_col="weight")
            gt_by_year[year] = sub

        for model in MODELS:
            orig = pd.read_parquet(os.path.join("data/benchmark", task_to_filename(model, task)))
            llm_lookup = orig.drop_duplicates(cond_vars)[cond_vars + ["llm_pred"]]

            for year in YEARS:
                sub = gt_by_year[year]
                available = sub is not None
                score = None
                if available:
                    res = sub.merge(llm_lookup, on=cond_vars, how="inner")
                    task_y = {**task, "dataset": f"data/clean/brfss_{year}.parquet"}
                    out = eval_hd(res, task_y, cache_dir="data/benchmark")
                    score = out["bench"].iloc[0]
                rows.append({"setting": "high", "outcome": v_out, "cond_vars": "_".join(cond_vars),
                             "model": model, "year": year, "score": score, "available": available})
                print(f"[high] {v_out:12s} {'_'.join(cond_vars):40s} {model:22s} {year}  "
                      f"{'—' if score is None else f'{score:6.2f}'}", flush=True)
    return rows


def main():
    eval_mod.cat_to_distr = _fast_cat_to_distr
    year_df = {y: pd.read_parquet(p) for y, p in YEAR_PARQUET.items()}

    rows = score_lowdim(year_df) + score_hd(year_df)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
