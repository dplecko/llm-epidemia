
import pandas as pd
import numpy as np
import sys
import os
import re
from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from utils.helpers import model_colors, task_to_filename
from utils.plot_helpers import name_and_sort
from utils.hd_helpers import hd_tasksize
from eval import build_eval_df
from task_spec import task_specs_hd
from plotnine import *

# models
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

# GPT likelihood analysis
fls = [s for s in os.listdir("data/benchmark") if re.search("PROB_gpt", s)]
all_t = [task_to_filename("", task_specs_hd[i]) for i in range(len(task_specs_hd))]

df = pd.DataFrame(all_t, columns = ["tasks"])
df["size"] = np.nan
df["dataset"] = ""
for i in tqdm(range(len(df))):
    df.loc[i, "size"] = hd_tasksize(task_specs_hd[i])[0]
    df.loc[i, "dataset"] = task_specs_hd[i]["dataset"]

# select all tasks with up to 250 queries
df["inc"] = df["size"] <= 250

# evaluate GPT 4.1 on select indices
gpt_idx = df[df["inc"] == True].index

task_sel = [task_specs_hd[i] for i in range(len(task_specs_hd)) if i in gpt_idx]
eval_gpt, eval_gptmap = build_eval_df(models + ["gpt-4.1", "o4-mini"], task_sel, prob = True)

df_gpt = eval_gpt.groupby(["model"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_gpt = name_and_sort(df_gpt)

# model mean baseline
mm, _ = build_eval_df(["model_mean"], task_sel)

plt_gpt = (ggplot(df_gpt, aes(x="Model", y="score", fill="Model")) +
       geom_point(size=0, color="white") +
       scale_fill_manual(values=model_colors) +
       annotate("rect", xmin=2.5, xmax=9.5, ymin=-2, ymax=42, alpha=0.2, fill="lightgray", color="gray") +
       annotate("text", x=8, y=44, label="Open-weights Models") +
       geom_col(color = "black") +
       labs(x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       geom_hline(yintercept = mm["score"].mean(), color = "orange", linetype = "dashed") +
       geom_text(aes(label=round(df_gpt["score"]).astype(int)), va="bottom", color="darkred", size=12, fontweight="bold") +
       geom_hline(yintercept = 100, color = "darkgreen", linetype = "dashed") +
       annotate("text", x=5.5, y=mm["score"].mean()+3, label="Mean-Baseline Score", color="orange", fontweight="bold") +
       annotate("text", x=1.5, y=95, label="Perfect Score", color="darkgreen", fontweight="bold") +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
             legend_position="none", axis_text_x=element_text(rotation=15)))

plt_gpt.save("data/plots/PROB_GPT.png", dpi=500, width=7.7, height=4)

# 0-score tasks for GPT-4.1
zscore_idx = eval_gpt[(eval_gpt["model"] == "gpt-4.1") & (eval_gpt["score"] == 0.0)]["task_id"].tolist()
rag_sel = np.array(gpt_idx)[zscore_idx]
rag_sel_fin = df.loc[rag_sel][df.loc[rag_sel, "size"] <= 150].index

# rag eval
eval_rag, _ = build_eval_df(["gpt-4.1_web"], 
                            [task_specs_hd[t] for t in range(len(task_specs_hd)) if t in rag_sel_fin],
                            prob = True)