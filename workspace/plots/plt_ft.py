
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from eval import build_eval_df
from utils.helpers import model_name, hd_taskname
from task_spec import task_specs, task_specs_hd
from plotnine import *

dataset = "brfss"
models = ["llama3_8b_instruct", "llama3_8b_instruct_sft", "mistral_7b_instruct", "mistral_7b_instruct_sft"]

# select all tasks which are NSDUH ("dataset" entry equals "data/nsduh.parquet")
tasks_dts = [t for t in task_specs + task_specs_hd if t["dataset"] == f"data/clean/{dataset}.parquet"]

eval_qa, _ = build_eval_df(models, tasks_dts, prob = False)
eval_lk, _ = build_eval_df(models, tasks_dts, prob = True)

eval_df = pd.concat([eval_qa, eval_lk]).reset_index()

for i in range(len(eval_df)):
    if eval_df.loc[i, "dim"] > 1:
        eval_df.loc[i, "task_name"] = hd_taskname(tasks_dts[eval_df.loc[i, "task_id"]])

# remove "NSDUH: " prefix from task names
eval_df["task_name"] = eval_df["task_name"].str.replace("NSDUH: ", "")
eval_df["task_name"] = eval_df["task_name"].str.replace("BRFSS: ", "")
eval_df["task_name"] = pd.Categorical(
    eval_df["task_name"], 
    categories=eval_df["task_name"].unique(), 
    ordered=True
)

### aggregated performance plots
eval_df["Model"] = model_name(eval_df["model"])
eval_df["Prompting"] = eval_df["prob"].apply(lambda x: "Likelihood" if x else "Q&A")
eval_df["Setting"] = eval_df["dim"].apply(lambda x: "Low-Dimensional" if x == 1 else "High-Dimensional")

# make sure Low-Dimensional is the first level in the Setting column
eval_df["Prompting"] = pd.Categorical(eval_df["Prompting"], categories=["Q&A", "Likelihood"], ordered=True)
eval_df["Setting"] = pd.Categorical(eval_df["Setting"], categories=["Low-Dimensional", "High-Dimensional"], ordered=True)

for model, col_scheme in zip(
    ["llama", "mistral"],
    [["#1f77b4", "#ff7f0e"],   # blue / orange
     ["#9467bd", "#d62728"]]  # purple / red
):
    subset_df = eval_df[eval_df["model"].str.contains(model, case=False)].reset_index()

    plt = (ggplot(subset_df, aes(x="task_name", y="score", fill="Model")) +
        geom_col(color="black", position="dodge") +
        scale_fill_manual(values=col_scheme) +
        labs(x="Task", y="Score") +
        theme_bw() + coord_cartesian(ylim=(0, 100)) +
        geom_text(aes(label=round(subset_df["score"]).astype(int)),
                  position=position_dodge(width=0.9),
                  va="bottom", size=6.5, color="darkred") +
        theme(axis_text_x=element_text(rotation=30, ha="right", size=5.5),
              panel_background=element_rect(fill="white"),
              plot_background=element_rect(fill="white"),
              legend_position="inside", legend_position_inside=(0.15, 0.85),
              legend_background=element_rect(color="black", fill="white")) +
        scale_x_discrete(expand=(0, 0)) +
        scale_y_continuous(expand=(0, 0)) +
        theme(plot_margin=0) +
        facet_grid('Prompting ~ Setting', scales="free_x")
    )
    plt.save(f"data/plots/{model}_sft_{dataset]}.png", dpi=300, width=10, height=5)


### model-level performance plots
# df_avg = eval_df.groupby(["Model", "Prompting"], as_index=False)["score"].mean()

# plt_avg = (ggplot(df_avg, aes(x="Model", y="score", fill="Model")) +
#            geom_col(color="black") +
#            labs(x="Model", y="Average Score") +
#            theme_bw() + coord_cartesian(ylim=(0, 100)) +
#            geom_text(aes(label=round(df_avg["score"]).astype(int)),
#                      va="bottom", size=10, color="darkred") +
#            theme(panel_background=element_rect(fill="white"),
#                  plot_background=element_rect(fill="white"),
#                  legend_position="none") + facet_grid(". ~ Prompting"))

# plt_avg.save("data/plots/llama3_8b_nsduh_avg.png", dpi=300, width=4, height=3)

### loss info plot
import json

rows = []
for path, name in [
    ("data/benchmark/llama_loss_history.json", "LLaMA3 8B"),
    ("data/benchmark/mistral_loss_history.json", "Mistral 7B")
]:
    with open(path) as f:
        records = json.load(f)
    for r in records:
        if "loss" in r:
            rows.append({"epoch": r["epoch"], "Loss": "Training", "loss_value": r["loss"], "Model": name})
        if "eval_loss" in r:
            rows.append({"epoch": r["epoch"], "Loss": "Evaluation", "loss_value": r["eval_loss"], "Model": name})

loss_info = pd.DataFrame(rows)

# early stopping = min eval loss
es_df = (loss_info.query("Loss == 'Evaluation'")
         .groupby("Model", as_index=False)
         .apply(lambda d: d.loc[d["loss_value"].idxmin(), ["epoch"]])
         .reset_index(drop=True))

# compute top y per model
ymax_df = (loss_info.groupby("Model", as_index=False)["loss_value"].max()
           .rename(columns={"loss_value": "y_top"}))

# merge so each model has its epoch + ymax
ann_df = es_df.merge(ymax_df, on="Model", how="left")
ann_df["y"] = ann_df["y_top"] * 0.6
ann_df["label"] = "Early Stopping"

plt_loss = (ggplot(loss_info, aes(x="epoch", y="loss_value", color="Loss")) +
            geom_line(size=1.25) +
            geom_vline(data=ann_df, mapping=aes(xintercept="epoch"), color="red", linetype="dashed") +
            geom_text(data=ann_df, mapping=aes(x="epoch-0.25", y="y", label="label"),
                      angle=90, ha="left", color="red", inherit_aes=False) +
            labs(x="Epoch", y="Loss Value") +
            theme_bw() +
            theme(legend_position="inside", legend_position_inside=(0.35, 0.85),
                  legend_background=element_rect(color="black", fill="white")) +
            facet_wrap("~Model", scales="free_y"))

plt_loss.save("data/plots/ft_loss_profile.png", dpi=300, width=10, height=3.5)


