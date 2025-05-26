
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

eval_ld, _ = build_eval_df(models, task_specs)
eval_hd, _ = build_eval_df(models, task_specs_hd)
df_lo = eval_ld.groupby(["model"]).agg(score=("score", "mean")).reset_index()
df_hi = eval_hd.groupby(["model"]).agg(score=("score", "mean")).reset_index()

df_cmb = pd.concat([df_lo, df_hi], keys=["low", "high"]).reset_index(level=0).rename(columns={"level_0": "Setting"})
df_cmb["Setting"] = df_cmb["Setting"].map({"low": "Low-Dimensional", "high": "High-Dimensional"})
df_cmb["Setting"] = pd.Categorical(df_cmb["Setting"], categories=["Low-Dimensional", "High-Dimensional"], ordered=True)
df_cmb["Model"] = model_name(df_cmb["model"])
mod_ord = df_cmb[df_cmb["Setting"] == "Low-Dimensional"].sort_values("score", ascending=False).index
df_cmb["Model"] = pd.Categorical(df_cmb["Model"], categories=df_cmb["Model"][mod_ord].unique(), ordered=True)
plt_lcomb = (ggplot(df_cmb, aes(x="Model", y="score", fill="Setting")) +
       geom_col(color = "black", position="dodge") +
       labs(x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       geom_text(aes(label=round(df_cmb["score"]).astype(int)), 
                 position=position_dodge(width=0.9),
                 va="bottom", color="darkred", size=12, fontweight="bold") +
       geom_hline(yintercept = 100, color = "darkgreen", linetype = "dashed") +
       annotate("text", x=1.5, y=95, label="Perfect Score", color="darkgreen", fontweight="bold") +
    #    scale_fill_manual(values=model_colors) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
             legend_position="inside", legend_position_inside=(0.7, 0.7),
             legend_background=element_rect(color="black", fill="white"),
             legend_margin=5))

plt_lcomb.save("data/plots/combined_leaderboard.png", dpi=300, width=5.5, height=3.3)
