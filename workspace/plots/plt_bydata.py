

import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from utils.helpers import model_name, dts_map, model_colors
from eval import build_eval_df
from task_spec import task_specs
from plotnine import *

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_dfq, _ = build_eval_df(models, task_specs)
eval_dfp, _ = build_eval_df(models, task_specs, prob=True)

eval_df = pd.concat([eval_dfq, eval_dfp])

# low-dimensional tasks: performance by dataset
df_bydata = eval_df.groupby(["model", "dataset", "prob"]).agg(
    score=("score", "mean")
).reset_index()
df_bydata["dataset"] = df_bydata["dataset"].map(dts_map)
# Calculate mean score per dataset
dataset_order = (df_bydata.groupby("dataset")["score"].mean()
                 .sort_values(ascending=False)
                 .index.tolist())
df_bydata["dataset"] = pd.Categorical(df_bydata["dataset"], categories=dataset_order, ordered=True)
df_bydata["Model"] = df_bydata["model"].apply(model_name)
df_bydata["Prompting"] = df_bydata["prob"].map(
    {False: "QA", True: "Likelihood"}
).astype("category").cat.set_categories(["QA", "Likelihood"], ordered=True)

plt_bydata = (ggplot(df_bydata, aes(x="dataset", y="score", fill = "Model")) +
         geom_col(position="dodge", color = "black") +
            labs(x="Dataset", y="Average Score", fill="Model") +
            scale_fill_manual(values=model_colors) +
            theme_bw() + coord_cartesian(ylim=(0, 100)) +
            guides(color=guide_legend(nrow=2)) +
            theme(
                panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
                legend_position="inside", legend_position_inside=(0.2, 0.8),
                legend_background=element_rect(color="black", fill="white"),
                legend_margin=5,
                legend_direction='horizontal',
                axis_text_x=element_text(rotation=30, hjust=0.8),
            ) + facet_wrap("~ Prompting"))
            #theme(axis.text.x = element_text(rotation=45, hjust=1))))

plt_bydata.save("data/plots/ld_bydataset.png", dpi=300, width=11, height=3.3)
