
from helpers import model_name
import pandas as pd
import plotnine as p9
import numpy as np
from metrics import weighted_corr
import os
import traceback
from helpers import task_to_filename

# plotting helpers
def name_and_sort(df):
    df["Model"] = df["model"].apply(model_name)
    df = df.sort_values("score", ascending=False)
    df["Model"] = pd.Categorical(df["Model"], categories=df["Model"], ordered=True)
    return df

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
                print(f"[ERROR] model={model}, task={task.get('name', i)}, error={e}")
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