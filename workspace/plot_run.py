
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
from helpers import model_name
from bench_eval import eval_task, eval_to_score, hd_corr_plot, hd_corr_df, build_eval_df
from task_spec import task_specs, task_specs_hd
from plotnine import *

# high-dimensional tasks: correlation
models = ["llama3_8b_instruct"]
plt_hdcor = hd_corr_plot([models], task_specs_hd)
cdf = hd_corr_df([models], task_specs_hd)
plt_hdcor.save("data/plots/hd_corr_plot.png", dpi=300, width=8, height=6)

# high-dimensional tasks: leaderboard
eval_df, eval_map = build_eval_df(models, task_specs_hd)
df_lead = eval_df.groupby(["model"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_lead["model"] = df_lead["model"].apply(model_name)
plt_lead = (ggplot(df_lead, aes(x="model", y="score")) +
       geom_col() +
       labs(title="High-Dimensional Leaderboard", x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white")))

plt_lead.save("data/plots/hd_leaderboard.png", dpi=300, width=8, height=6)

# performance by dimension
df_bydim = eval_df.groupby(["model", "dim"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_bydim["model"] = df_bydim["model"].apply(model_name)
plt_bydim = (ggplot(df_bydim, aes(x="dim", y="score", fill = "model")) +
       geom_col(position="dodge", color = "black") +
       labs(title="Performance by Dimension", x="Dimension", y="Average Score", fill="Model") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white")))

plt_bydim.save("data/plots/hd_bydim.png", dpi=300, width=8, height=6)

# pd.read_parquet('data/benchmark/llama3_8b_instruct_brfss_diabetes_sex_race.parquet')

# eval_hd with new lgbm boostrap
eval_task("llama3_8b_instruct", task_specs_hd[0])

eval_task("llama3_8b_instruct", task_specs[20])