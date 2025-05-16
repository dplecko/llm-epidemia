
import pandas as pd
import sys, os
import numpy as np
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

def name_and_sort(df):
    # df = eval_hd.groupby("model").agg(
    #     score=("score", "mean")
    # ).reset_index()
    df["Model"] = df["model"].apply(model_name)
    df = df.sort_values("score", ascending=False)
    df["Model"] = pd.Categorical(df["Model"], categories=df["Model"], ordered=True)
    return df


# low-dimensional leaderboard
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_df, eval_map = build_eval_df(models, task_specs)

df_low = eval_df.groupby(["model"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
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


# low-dimensional tasks: performance by dataset
df_bydata = eval_df.groupby(["model", "dataset"]).agg(
    score=("score", "mean")
).reset_index()
dts_map = {
    "nhanes": "NHANES",
    "gss": "GSS",
    "brfss": "BRFSS",
    "nsduh": "NSDUH",
    "acs": "ACS",
    "edu": "IPEDS",
    "fbi_arrests": "FBI Arrests",
    "labor": "BLS",
    "meps": "MEPS",
    "scf": "SCF",
}
df_bydata["dataset"] = df_bydata["dataset"].map(dts_map)
# Calculate mean score per dataset
dataset_order = (df_bydata.groupby("dataset")["score"].mean()
                 .sort_values(ascending=False)
                 .index.tolist())
df_bydata["dataset"] = pd.Categorical(df_bydata["dataset"], categories=dataset_order, ordered=True)
df_bydata["Model"] = df_bydata["model"].apply(model_name)
plt_bydata = (ggplot(df_bydata, aes(x="dataset", y="score", fill = "Model")) +
         geom_col(position="dodge", color = "black") +
            labs(x="Dataset", y="Average Score", fill="Model") +
            scale_fill_manual(values=model_colors) +
            theme_bw() + coord_cartesian(ylim=(0, 100)) +
            guides(color=guide_legend(nrow=2)) +
            theme(
                panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
                legend_position="inside", legend_position_inside=(0.85, 0.8),
                legend_background=element_rect(color="black", fill="white"),
                legend_margin=5,
                legend_direction='horizontal',
                axis_text_x=element_text(rotation=30, hjust=0.8),
            ))
            #theme(axis.text.x = element_text(rotation=45, hjust=1))))
plt_bydata.save("data/plots/ld_bydataset.png", dpi=300, width=5.5, height=3.3)

# high-dimensional tasks: correlation
plt_hdcor = hd_corr_plot(models, task_specs_hd)
cdf = hd_corr_df([models], task_specs_hd)
plt_hdcor.save("data/plots/hd_corr_plot.png", dpi=300, width=8, height=6)

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

# low-dimensional performance by dataset

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

df_cmb = pd.concat([df_low, df_lead], keys=["low", "high"]).reset_index(level=0).rename(columns={"level_0": "Setting"})
df_cmb["Setting"] = df_cmb["Setting"].map({"low": "Low-Dimensional", "high": "High-Dimensional"})
df_cmb["Setting"] = pd.Categorical(df_cmb["Setting"], categories=["Low-Dimensional", "High-Dimensional"], ordered=True)
df_cmb["Model"] = pd.Categorical(df_cmb["Model"], categories=df_low["Model"], ordered=True)
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