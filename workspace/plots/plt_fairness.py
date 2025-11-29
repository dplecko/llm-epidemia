

import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from eval import build_eval_df
from utils.helpers import model_name
from task_spec import task_specs, task_specs_hd
from plotnine import *

def task_vars(task):
    # if full list provided
    if task.get("variables") is not None:
        return task.get("variables")

    # otherwise build [v_out] + v_cond
    v_out = task.get("v_out")
    v_cond = task.get("v_cond") or []

    # ensure v_out is always a list
    if not isinstance(v_out, list):
        v_out = [v_out]

    return v_out + v_cond

def is_sens(task):
    vars = task_vars(task)
    if "sex" in vars or "race" in vars:
        return True
    return False

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

eval_lo = []
eval_hi = []
for p in (False, True):
    lo = build_eval_df(models, task_specs, prob=p)[0]
    lo["Sensitive"] = False
    for i in range(len(lo)):
        lo.loc[i, "Sensitive"] = is_sens(task_specs[lo.loc[i, "task_id"]])
    eval_lo.append(lo)
    
    hi = build_eval_df(models, task_specs_hd, prob=p)[0]
    hi["Sensitive"] = False
    for i in range(len(hi)):
        hi.loc[i, "Sensitive"] = is_sens(task_specs_hd[hi.loc[i, "task_id"]])
    eval_hi.append(hi)

df = pd.concat(eval_lo + eval_hi, ignore_index=True)
df = df.groupby(["model", "prob", "Sensitive"]).agg(score=("score", "mean")).reset_index()

df["Model"] = model_name(df["model"])
mod_ord = (
    df[(df["prob"] == False)]
    .sort_values("score", ascending=False)
    .index
)
df["Model"] = pd.Categorical(df["Model"], categories=df["Model"][mod_ord].unique(), ordered=True)

# precompute labels
df = df.reset_index(drop=True)
df["lab"] = df["score"].round().astype(int)
df["Sensitive"] = df["Sensitive"].map({False: "No", True: "Yes"}).astype("category").cat.set_categories(["No", "Yes"], ordered=True)
df["Prompting"] = df["prob"].map({False: "QA", True: "Likelihood"}).astype("category").cat.set_categories(["QA", "Likelihood"], ordered=True)

plt_sens = (
    ggplot(df, aes(x="Model", y="score", fill="Sensitive"))
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

plt_sens
plt_sens.save("data/plots/fairness_impact.png", dpi=300, width=11, height=3.3)
