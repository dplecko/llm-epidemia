
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from eval import build_eval_df
from utils.helpers import model_name
from task_spec import task_specs, task_specs_hd
from plotnine import *

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

eval_lo = []
eval_hi = []
for p in (False, True):
    eval_lo.append(build_eval_df(models, task_specs, prob=p)[0])
    eval_hi.append(build_eval_df(models, task_specs_hd, prob=p)[0])

df_lo = pd.concat(eval_lo, ignore_index=True)
df_hi = pd.concat(eval_hi, ignore_index=True)

df_lo = df_lo.groupby(["model", "prob"]).agg(score=("score", "mean")).reset_index()
df_hi = df_hi.groupby(["model", "prob"]).agg(score=("score", "mean")).reset_index()

df_cmb = pd.concat([df_lo, df_hi], keys=["low", "high"]).reset_index(level=0).rename(columns={"level_0": "Setting"})
df_cmb["Setting"] = df_cmb["Setting"].map({"low": "Low-Dimensional", "high": "High-Dimensional"})
df_cmb["Setting"] = pd.Categorical(df_cmb["Setting"], categories=["Low-Dimensional", "High-Dimensional"], ordered=True)
df_cmb["Model"] = model_name(df_cmb["model"])
mod_ord = (
    df_cmb[(df_cmb["Setting"] == "Low-Dimensional") &
           (df_cmb["prob"] == False)]
    .sort_values("score", ascending=False)
    .index
)
df_cmb["Model"] = pd.Categorical(df_cmb["Model"], categories=df_cmb["Model"][mod_ord].unique(), ordered=True)

# precompute labels
df_cmb = df_cmb.reset_index(drop=True)
df_cmb["lab"] = df_cmb["score"].round().astype(int)
df_cmb["Prompting"] = df_cmb["prob"].map({False: "QA", True: "Likelihood"}).astype("category").cat.set_categories(["QA", "Likelihood"], ordered=True)

plt_lcomb = (
    ggplot(df_cmb, aes(x="Model", y="score", fill="Setting"))
    + geom_col(color="black", position="dodge")
    + geom_text(
        aes(label="lab"),
        position=position_dodge(width=0.9),
        va="bottom",
        color="darkred",
        size=12,
        fontweight="bold"
    )
    + labs(x="Model", y="Average Score")
    + theme_bw()
    + coord_cartesian(ylim=(0, 100))
    + geom_hline(yintercept=100, color="darkgreen", linetype="dashed")
    + annotate("text", x=1.5, y=95, label="Perfect Score",
               color="darkgreen", fontweight="bold")
    + theme(
        panel_background=element_rect(fill="white"),
        plot_background=element_rect(fill="white"),
        legend_position="inside",
        legend_position_inside=(0.35, 0.7),
        legend_background=element_rect(color="black", fill="white"),
        legend_margin=5
    )
    + facet_wrap("~ Prompting")
)

plt_lcomb
plt_lcomb.save("data/plots/combined_leaderboard.png", dpi=300, width=11, height=3.3)
