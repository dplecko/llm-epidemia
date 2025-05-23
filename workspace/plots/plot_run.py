
import pandas as pd
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from helpers import model_name, dts_map
from bench_eval import hd_corr_plot, hd_corr_df, build_eval_df
from task_spec import task_specs, task_specs_hd
from plotnine import *

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

# likelihood plots
models = ["llama3_70b_instruct", "llama3_8b_instruct", "mistral_7b_instruct", "phi4",
          "gemma3_27b_instruct", "deepseek_7b_chat", "deepseekR1_32b"]
eval_prob, eval_probmap = build_eval_df(models, task_specs_hd, prob = True)
df_prob = eval_prob.groupby(["model"]).agg(
    score=("score", "mean"),
    # sd_cor=("correlation", "std")
).reset_index()
df_prob = name_and_sort(df_prob)

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

# response distribution for models
# fls = os.listdir("data/benchmark")
# res = pd.DataFrame()
# for f in tqdm(fls):
#     if "PROB" in f:
#         df = pd.read_parquet("data/benchmark/" + f)
#         df = df[["llm_pred"]]
#         df["model"] = "_".join(f.split("_")[1:4])
#         df["dataset"] = f.split("_")[4]
#         df["task"] = f
#         res = pd.concat([res, df])
#         # print("\n", f, "\n")
#         # print(df["llm_pred"].value_counts())

# plt_distr = (ggplot(res, aes(x='llm_pred')) + geom_histogram(bins=22) +
#     theme_bw() + facet_wrap(["model"]))

# plt_distr.save("data/plots/prob_distr.png", dpi=300, width=5.5, height=3.3)

# GPT likelihood analysis
fls = [s for s in os.listdir("data/benchmark") if re.search("PROB_gpt", s)]

gpt_t = [x.replace("PROB_gpt-4.1", "") for x in fls]
all_t = [task_to_filename("", task_specs_hd[i]) for i in range(len(task_specs_hd))]

df = pd.DataFrame(all_t, columns = ["tasks"])
df["has_gpt"] = False
df["size"] = np.nan
df["dataset"] = ""
for i in tqdm(range(len(df))):
    df.loc[i, "has_gpt"] = df.loc[i, "tasks"] in gpt_t
    df.loc[i, "size"] = hd_tasksize(task_specs_hd[i])[0]
    df.loc[i, "dataset"] = task_specs_hd[i]["dataset"]

ggplot(df, aes(x = "tasks", y = "size", fill = "has_gpt")) + geom_col()

np.quantile(df["size"], q = [0.5, 0.75, 0.9])

prop = np.cumsum(sorted(df["size"].values)) / sum(df["size"])

df[df["has_gpt"] == True]["size"].sum()

df["inc"] = df["size"] <= 250
(df["inc"] == True).sum()
df[df["inc"] == True]["dataset"].value_counts()
df[df["inc"] == True]["size"].sum()

all_pred = pd.DataFrame()
for f in tqdm(fls):
    preds = pd.read_parquet("data/benchmark/" + f)
    preds = preds[["llm_pred"]]
    all_pred = pd.concat([all_pred, preds])


# plt_distr = (ggplot(all_pred, aes(x='llm_pred')) + geom_histogram(bins=22, fill="red",color="black") +
#     theme_bw())

# plt_distr

# evaluate GPT 4.1 on select indices
gpt_idx = [0,  1,  2,  3,  4,  6,  7, 13, 14, 15, 16, 17, 18, 19, 20, 26, 27, 28, 29, 31, 32, 33, 39, 40, 42,
  44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 68, 69, 70, 72, 73, 74, 75, 76,
  77, 78, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91]

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

plt_gpt

plt_gpt.save("data/plots/PROB_GPT.png", dpi=500, width=7.7, height=4)

print(eval_gpt[eval_gpt["model"] == "gpt-4.1"])

# poor performance tasks
zscore_idx = eval_gpt[(eval_gpt["model"] == "gpt-4.1") & (eval_gpt["score"] == 0.0)]["task_id"].tolist()
rag_sel = np.array(gpt_idx)[zscore_idx]
rag_sel_fin = df.loc[rag_sel][df.loc[rag_sel, "size"] <= 150].index

# rag eval
eval_rag, _ = build_eval_df(["gpt-4.1_web"], 
                            [task_specs_hd[t] for t in range(len(task_specs_hd)) if t in rag_sel_fin],
                            prob = True)

# heatmap
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_heat, _ = build_eval_df(models, task_specs_hd)
ordr = eval_heat.groupby(["model"]).agg(score=("score", "mean"))["score"]
eval_heat["model"] = pd.Categorical(eval_heat["model"], categories=ordr.sort_values(ascending=True).index.values, ordered=True)

hmap = (ggplot(eval_heat, aes(y = "model", x = "task_id", fill="score")) + 
    geom_tile() + 
    theme_bw() +
    geom_text(aes(label="round(score)")) +
    # theme(axis_text_y=element_text(rotation = 15)) +
    scale_fill_cmap(cmap_name="viridis"))

hmap.save("data/plots/heatmap.png", dpi=500, width=25, height=5)
