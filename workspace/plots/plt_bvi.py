
from workspace.common import *

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

models_base = [re.sub(r"(_instruct|_chat)", "", x) for x in models]

eval_df, eval_map = build_eval_df(models + models_base, task_specs_hd)
eval_df["Model"] = model_name(eval_df["model"])
eval_df["Instruction Tuned"] = eval_df["model"].str.contains(r"(_instruct|_chat)")

# plot one: side-by-side performance
df_bvi = eval_df.groupby(["Model", "model", "Instruction Tuned"]).agg(score=("score", "mean")).reset_index()

mod_ord = (df_bvi.groupby("Model")["score"].mean().sort_values(ascending=False).index.tolist())
df_bvi["Model"] = pd.Categorical(df_bvi["Model"], categories=mod_ord, ordered=True)

plt_bvi = (ggplot(df_bvi, aes(x="Model", y="score", fill="Instruction Tuned")) +
       geom_col(color = "black", position="dodge") +
       labs(x="Model", y="Average Score") +
       theme_bw() + coord_cartesian(ylim=(0, 100)) +
    #    scale_fill_manual(name="Model Type", values={"Base": "#1f77b4", "Instruct": "#ff7f0e"},
    #                      breaks=["Base", "Instruct"]) +
       geom_text(aes(label=round(df_bvi["score"]).astype(int)), 
                 position=position_dodge(width=0.9),
                 va="bottom", color="darkred", size=12, fontweight="bold") +
       geom_hline(yintercept = 100, color = "darkgreen", linetype = "dashed") +
       annotate("text", x=1.5, y=95, label="Perfect Score", color="darkgreen", fontweight="bold") +
       #    scale_fill_manual(values=model_colors) +
       theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"),
             legend_position="inside", legend_position_inside=(0.7, 0.7),
             legend_background=element_rect(color="black", fill="white"),
             legend_margin=5))

plt_bvi.save("data/plots/base_vs_instruct.png", dpi=300, width=5.5, height=3.3)

df_dep = pd.merge(eval_df[eval_df["Instruction Tuned"] == False], 
                  eval_df[eval_df["Instruction Tuned"] == True], on = ["Model", "task_id"])
plt_dep = (
    ggplot(df_dep, aes(x="score_x", y = "score_y")) +
    geom_point() +
    labs(x = "Base Model Score", y = "Instruct Model Score") +
    coord_cartesian(xlim = (0, 100), ylim=(0, 100)) +
    geom_abline(slope = 1, intercept = 0, linetype="dashed", color = "orange") +
    theme_bw() +
    facet_wrap(["Model"], ncol = 5)
)

plt_dep.save("data/plots/base_vs_instruct_bytask.png", dpi=300, width=15, height=3.5)
