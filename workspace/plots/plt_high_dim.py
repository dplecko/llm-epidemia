
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from utils.helpers import model_name, model_colors
from utils.plot_helpers import name_and_sort
from eval import build_eval_df
from task_spec import task_specs_hd
from plotnine import *

# models
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

# high-dimensional tasks: leaderboard
eval_hd, eval_hdmap = build_eval_df(models, task_specs_hd)
df_lead = eval_hd.groupby(["model"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_lead = name_and_sort(df_lead)
plt_lead = (ggplot(df_lead, aes(x="Model", y="score", fill="Model")) +
       geom_col(color = "black") +
       labs(title="High-Dimensional Setting Leaderboard", x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       geom_text(aes(label=round(df_lead["score"]).astype(int)), va="bottom", color="darkred", size=12, fontweight="bold") +
       geom_hline(yintercept = 100, color = "darkgreen", linetype = "dashed") +
       annotate("text", x=1.5, y=95, label="Perfect Score", color="darkgreen", fontweight="bold") +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
             legend_position="none", 
             axis_title_x=element_text(margin={"t": 40, "r": 0, "b": 0, "l": 0}),
             axis_text_x=element_text(margin={"t": 0, "r": 0, "b": 50, "l": 0})) +
       scale_fill_manual(values=model_colors))

plt_lead.save("data/plots/hd_leaderboard.svg", dpi=300, width=5.5, height=4)

# performance by dimension
df_bydim = eval_hd.groupby(["model", "dim"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_bydim["Model"] = df_bydim["model"].apply(model_name)
df_bydim["dim"] = df_bydim["dim"].astype(str)
df_bydim["dim"] = "d = " + df_bydim["dim"]
plt_bydim = (ggplot(df_bydim, aes(x="dim", y="score", fill = "Model")) +
       geom_col(position="dodge", color = "black") +
       labs(x="Dimension", y="Average Score", fill="Model") +
       scale_fill_manual(values=model_colors) +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
             legend_position="none", axis_text_x=element_text(rotation=30, hjust=0.8, size=12)))

plt_bydim.save("data/plots/hd_bydim.png", dpi=300, width=5.5, height=3.3)
