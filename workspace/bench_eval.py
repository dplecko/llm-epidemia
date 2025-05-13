
import pandas as pd
import numpy as np
import plotnine as p9
import re
import sys
import os
import json
sys.path.append(os.path.abspath("workspace"))
from metrics import ks_w, cat_to_distr
from helpers import task_to_filename, weighted_corr, weighted_L1
from hd_helpers import bootstrap_lgbm

def eval_cat(res, model_name, mode, dataset, v1, v2, levels):
    nbins = len(levels)
    lvl_names = levels
    rows = []
    distr_rows = []
    cond_wghs = []
    score_c = []
    worst_c = []
    best_c = []

    for r in res:
        
        cond_wghs.append(r["total_weight"])
        if len(r["model_vals"]) == len(levels):
            n_mc = r["n_data"]
        else:
            n_mc = len(r["model_vals"]) 
        val_true = np.array([levels.index(v) for v in r["true_vals"]], dtype=int)
        wgh_true = np.array(r.get("true_weights", [1.0] * len(val_true)))
        
        val_mod = np.array([levels.index(m) for m in r["model_vals"]])
        wgh_mod = np.array(r.get("model_weights", [1.0] * len(val_true)))

        distr_true = cat_to_distr(val_true, wgh_true, nbins)
        distr_mod = cat_to_distr(val_mod, wgh_mod, nbins)

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
        best_cboot = []
        for _ in range(100):
            idx = np.random.choice(len(val_true), size=n_mc, p=wgh_true / wgh_true.sum(), replace=True)
            val_bt = val_true[idx]
            distr_bt = cat_to_distr(val_bt, None, nbins)
            best_cboot.append(np.abs(distr_bt - distr_true).sum())
        best_c.append(best_cboot)

        # worst error for this conditioning set
        worst_c.append(np.abs(np.ones(nbins) / nbins - distr_true).sum())
        
        # score for the conditioning set
        score_c.append(np.abs(distr_true - distr_mod).sum()) 

        rows.append({
            "cond": r["condition"]
        })

    worst_err = np.sum(np.array(worst_c) * np.array(cond_wghs)) / np.sum(cond_wghs)
    score = np.sum(np.array(score_c) * np.array(cond_wghs)) / np.sum(cond_wghs)

    best_cx = np.array(best_c)
    best_cx = np.average(best_cx, axis=0, weights=np.array(cond_wghs))
    best_err = np.quantile(best_cx, 0.975)

    # only scores are essentially 0 and 100!
    if worst_err < best_err :
        worst_err = best_err + 0.0001

    bench = (score - worst_err) / (best_err - worst_err)
    bench = 100 * max(min(bench, 1), 0)

    df = pd.DataFrame(rows)
    df["bench"] = bench
    df.attrs["distr"] = pd.DataFrame(distr_rows)
    return df

def hd_best_err(res, task):

    return 0
    cond_vars = task["v_cond"]
    out_var = task["v_out"]
    cond_vars_str = "_".join(task_spec["v_cond"])
    dataset_name = task_spec['dataset'].split('/')[-1].split('.')[0]
    file_name = f"best_err_{dataset_name}_{task_spec['v_out']}_{cond_vars_str}.txt"
    if os.path.exists(os.path.join("data", "benchmark", file_name)):
        with open(os.path.join("data", "benchmark", file_name), "r") as f:
            best_err = float(f.read())
        return best_err
    else:
        boot_mat = bootstrap_lgbm(res[cond_vars + [out_var] + ["weight"]], out_var, cond_vars, wgh_col="weight")
        for i in range(boot_mat.shape[1]):
            best_err.append(weighted_L1(boot_mat[:, i], res["lgbm_pred"], res["weight"]))
        best_err = np.quantile(best_err, 0.975)
        with open(os.path.join("data", "benchmark", file_name), "w") as f:
            f.write(str(best_err))
        return best_err

def eval_hd(res, task):

    # get the best error
    best_err = hd_best_err(res, task)

    # get the true score
    score = weighted_L1(res["llm_pred"], res["lgbm_pred"], res["weight"])
    
    # get the worst error
    b1 = weighted_L1(np.zeros(len(res)), res["lgbm_pred"], res["weight"])
    b2 = weighted_L1(np.ones(len(res)), res["lgbm_pred"], res["weight"])
    b3 = weighted_L1(np.ones(len(res)) * 0.5, res["lgbm_pred"], res["weight"])
    worst_err = min(b1, b2, b3)

    # only scores are essentially 0 and 100!
    if worst_err < best_err :
        worst_err = best_err + 0.0001

    # compute the benchmark normalized score
    bench = (score - worst_err) / (best_err - worst_err)
    bench = 100 * max(min(bench, 1), 0)
    df_eval = pd.DataFrame({
        "bench": [bench],
        "llm_pred": [res["llm_pred"].tolist()],
        "lgbm_pred": [res["lgbm_pred"].tolist()],
        "weight": [res["weight"].tolist()],
    })
    return df_eval
     
def eval_to_score(df):
    return df["bench"].mean()

def dat_name_clean(path):
    base = path.split("/")[-1]
    return re.sub(r"\.parquet$", "", base)

def eval_task(model_name, task):

    mode = "logits"
    dataset = dat_name_clean(task["dataset"])

    fname = task_to_filename(model_name, task)
    path = os.path.join("data", "benchmark", fname)

    if "json" in path:
        with open(path, "r") as f:
            res = json.load(f)

        v1 = task["variables"][0]
        v2 = task["variables"][1] if len(task["variables"]) > 1 else None
        levels = pd.read_parquet(task["dataset"])[v1].unique().tolist()
        return eval_cat(res, model_name, mode, dataset, v1, v2, levels)
    elif "parquet" in path:
        res = pd.read_parquet(path)
        return eval_hd(res, task)

def build_eval_df(models, tasks):
    rows = []
    eval_map = {}
    for model in models:
        for i, task in enumerate(tasks):
            try:
                df_eval = eval_task(model, task)
                score = eval_to_score(df_eval)
            except Exception as e:
                print(f"[ERROR] model={model}, task={task.get('name', i)}")
                traceback.print_exc()
                df_eval, score = None, None
            
            eval_map[i] = df_eval
            rows.append({
                "model": model,
                "task_id": i,
                "task_name": task.get("name", i),
                "dim": len(task["v_cond"]) if "v_cond" in task else 1,
                "score": score,
            })

    return pd.DataFrame(rows), eval_map

# high-dimensional correlation plots functionality
def hd_corr_df(models, tasks):

    rows = []
    for model in models:
        for i, task in enumerate(tasks):
            try:
                fname = task_to_filename(model, task)
                df = pd.read_parquet(os.path.join("data", "benchmark", fname))
                df["llm_pred"] += np.random.uniform(-0.001, 0.001, size=len(df))
                cor_ll = weighted_corr(df["llm_pred"], df["lgbm_pred"], df["weight"])
                dim = len(task["v_cond"])
            except Exception as e:
                print(f"[ERROR] model={model}, task={task.get('name', i)}")
                traceback.print_exc()
                df, cor_ll, fname, dim = None, None, None, None
            
            rows.append({
                "model": model,
                "task_id": i,
                "task_name": fname,
                "dim": dim,
                "correlation": cor_ll,
            })
    
    return pd.DataFrame(rows)

def hd_corr_plot(models, tasks):
    df = hd_corr_df(models, tasks)
    df_agg = df.groupby(["model", "dim"]).agg(
        avg_cor=("correlation", "mean"),
        sd_cor=("correlation", "std")
    ).reset_index()

    plt = (p9.ggplot(df_agg, p9.aes(x="dim", y="avg_cor", color="model")) +
       p9.geom_point() +
       p9.geom_errorbar(p9.aes(ymin="avg_cor - sd_cor", ymax="avg_cor + sd_cor"), width=0.2) +
       p9.labs(title="High-Dimensional Correlation", x="Number of Dimensions", y="Avg. Corr.") +
       p9.theme_bw() +
       p9.coord_cartesian(ylim=(-1, 1)) +
       p9.theme(panel_background=p9.element_rect(fill="white"), plot_background=p9.element_rect(fill="white")))
    
    return plt