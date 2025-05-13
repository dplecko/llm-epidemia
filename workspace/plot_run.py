
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
from helpers import model_name
from bench_eval import eval_task, eval_to_score, hd_corr_plot, hd_corr_df, build_eval_df
from task_spec import task_specs, task_specs_hd
from plotnine import *

model_colors = {
    "LLama3 8B": "#1f77b4",
    "LLama3 70B": "#ff7f0e",
    "Mistral 7B": "#2ca02c",
    "DeepSeek 7B": "#17becf",
    "Phi4": "#d62728",
    "Gemma3 27B": "#9467bd",
}

# low-dimensional leaderboard
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", "gemma3_27b_instruct"]
eval_df, eval_map = build_eval_df(models, task_specs)
df_low = eval_df.groupby(["model"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_low["model"] = df_low["model"].apply(model_name)
plt_low = (ggplot(df_low, aes(x="model", y="score", fill="model")) +
       geom_col(color = "black") +
       labs(title="Low-Dimensional Leaderboard", x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white")) +
       scale_fill_manual(values=model_colors))
plt_low.save("data/plots/ld_leaderboard.png", dpi=300, width=8, height=6)

# high-dimensional tasks: correlation
models = ["llama3_8b_instruct"]
plt_hdcor = hd_corr_plot([models], task_specs_hd)
cdf = hd_corr_df([models], task_specs_hd)
plt_hdcor.save("data/plots/hd_corr_plot.png", dpi=300, width=8, height=6)

# high-dimensional tasks: leaderboard
eval_hd, eval_hdmap = build_eval_df(models, task_specs_hd)
df_lead = eval_hd.groupby(["model"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_lead["model"] = df_lead["model"].apply(model_name)
plt_lead = (ggplot(df_lead, aes(x="model", y="score")) +
       geom_col(color = "black") +
       labs(title="High-Dimensional Leaderboard", x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white")))

plt_lead.save("data/plots/hd_leaderboard.png", dpi=300, width=8, height=6)

# low-dimensional performance by dataset

# performance by dimension
df_bydim = eval_hd.groupby(["model", "dim"]).agg(
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
# eval_task("llama3_8b_instruct", task_specs_hd[0])
# eval_task("llama3_8b_instruct", task_specs[20])

path = 'data/benchmark/gemma3_27b_instruct_acs_employment_status_sex.json'
with open(path, "r") as f:
    res = json.load(f)
