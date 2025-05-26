
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