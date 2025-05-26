
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from utils.plot_helpers import name_and_sort
from utils.helpers import model_colors
from eval import build_eval_df
from task_spec import task_specs_hd
from plotnine import *

# likelihood plots
models = ["llama3_70b_instruct", "llama3_8b_instruct", "mistral_7b_instruct", "phi4",
          "gemma3_27b_instruct", "deepseek_7b_chat", "deepseekR1_32b"]
eval_prob, eval_probmap = build_eval_df(models, task_specs_hd, prob = True)
df_prob = eval_prob.groupby(["model"]).agg(score=("score", "mean")).reset_index()


# model mean baseline
mm, _ = build_eval_df(["model_mean"], task_specs_hd)

plt_prob = (ggplot(df_prob, aes(x="Model", y="score", fill="Model")) +
       geom_col(color = "black") +
       labs(x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       geom_text(aes(label=round(df_prob["score"]).astype(int)), va="bottom", color="darkred", size=12, fontweight="bold") +
       geom_hline(yintercept = 100, color = "darkgreen", linetype = "dashed") +
       geom_hline(yintercept = mm["score"].mean(), color = "orange", linetype = "dashed") +
       annotate("text", x=5.5, y=mm["score"].mean()+3, label="Mean-Baseline Score", color="orange", fontweight="bold") +
       annotate("text", x=1.5, y=95, label="Perfect Score", color="darkgreen", fontweight="bold") +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
             legend_position="none", axis_text_x=element_text(rotation=10)) +
       scale_fill_manual(values=model_colors))

plt_prob.save("data/plots/PROB_leaderboard.png", dpi=300, width=6.6, height=3.3)

# side-by-side likelihood vs. QA prompting
eval_qa, _ = build_eval_df(models, task_specs_hd, prob = False)
df_qa = eval_qa.groupby(["model"]).agg(score=("score", "mean")).reset_index()
df_qa = name_and_sort(df_qa)

df_prompt = pd.concat([df_prob, df_qa], keys=["Likelihood", "Q&A"]).reset_index(level=0).rename(columns={"level_0": "Prompting"})
df_prompt["Prompting"] = df_prompt["Prompting"].map({"Likelihood": "Likelihood", "Q&A": "Q&A"})

df_prompt = df_prompt[df_prompt["model"] != "deepseekR1_32b"]
df_prompt["Model"] = pd.Categorical(df_prompt["Model"], categories = df_prompt["Model"].unique(), ordered=True)
plt_prompt = (ggplot(df_prompt, aes(x="Model", y="score", fill="Prompting")) +
       geom_col(color = "black", position="dodge") +
       labs(x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
       geom_text(aes(label=round(df_prompt["score"]).astype(int)), 
                 position=position_dodge(width=0.9),
                 va="bottom", color="darkred", size=12, fontweight="bold") +
       geom_hline(yintercept = 100, color = "darkgreen", linetype = "dashed") +
       annotate("text", x=1.5, y=95, label="Perfect Score", color="darkgreen", fontweight="bold") +
    #    scale_fill_manual(values=model_colors) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
             legend_position="inside", legend_position_inside=(0.7, 0.7),
             legend_background=element_rect(color="black", fill="white"),
             legend_margin=5))

plt_prompt.save("data/plots/qa_vs_likelihood.png", dpi=300, width=5.5, height=3.3)
