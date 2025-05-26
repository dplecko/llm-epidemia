
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from eval import build_eval_df
from utils.plot_helpers import name_and_sort
from utils.helpers import model_colors
from task_spec import task_specs
from plotnine import *

# low-dimensional leaderboard
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_df, eval_map = build_eval_df(models, task_specs)

df_low = eval_df.groupby(["model"]).agg(score=("score", "mean")).reset_index()
df_low = name_and_sort(df_low)
plt_low = (ggplot(df_low, aes(x="Model", y="score", fill="Model")) +
       geom_col(color = "black") +
       labs(title="Low-Dimensional Leaderboard", x="Model", y="Average Score") +
       geom_text(aes(label=round(df_low["score"]).astype(int)), va="bottom", color="darkred", size=12, fontweight="bold") +
       geom_hline(yintercept = 100, color = "darkgreen", linetype = "dashed") +
       annotate("text", x=1.5, y=97, label="Perfect Score", color="darkgreen", fontweight="bold") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white")) +
       scale_fill_manual(values=model_colors))

plt_low.save("data/plots/ld_leaderboard.png", dpi=300, width=8, height=6)
