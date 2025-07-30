
import pandas as pd
import plotly.express as px
from workspace.utils.helpers import model_name, dts_map, model_colors
from workspace.eval import build_eval_df
from task_spec import task_specs

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_df, eval_map = build_eval_df(models, task_specs)

# Aggregate
df = eval_df.groupby(["model", "dataset"]).agg(score=("score", "mean")).reset_index()
df["dataset"] = df["dataset"].map(dts_map)
df["Model"] = df["model"].apply(model_name)
df["Color"] = df["Model"].map(model_colors)

# Order datasets by mean score
dataset_order = df.groupby("dataset")["score"].mean().sort_values(ascending=False).index.tolist()
df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)

# Plot
fig = px.bar(
    df, x="dataset", y="score", color="Model",
    barmode="group", text_auto=".0f", color_discrete_map=model_colors
)
fig.update_layout(
    title="Average Score by Dataset",
    xaxis_title="Dataset",
    yaxis_title="Average Score",
    yaxis=dict(range=[0, 100]),
    plot_bgcolor="#0d1b2a",
    paper_bgcolor="#0d1b2a",
    font=dict(color="white"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_traces(marker_line_color="black", marker_line_width=0.5)

fig.write_html("www/img/interactive_ld_bydataset.html", include_plotlyjs="cdn")
