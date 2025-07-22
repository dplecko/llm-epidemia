
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from utils.helpers import model_name, model_colors
from eval import build_eval_df
from task_spec import task_specs_hd

# models and data
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_hd, _ = build_eval_df(models, task_specs_hd)

# aggregate by dimension
df_bydim = eval_hd.groupby(["model", "dim"]).agg(score=("score", "mean")).reset_index()
df_bydim["Model"] = df_bydim["model"].apply(model_name)
df_bydim["Dimension"] = "d = " + df_bydim["dim"].astype(str)

# plotly bar chart
fig = px.bar(
    df_bydim,
    x="Dimension",
    y="score",
    color="Model",
    barmode="group",
    text_auto=".0f",
    color_discrete_map=model_colors,
    labels={"score": "Average Score"},
    title="Performance by Dimension"
)

fig.update_layout(
    plot_bgcolor="#0d1b2a",
    paper_bgcolor="#0d1b2a",
    font=dict(color="white"),
    xaxis=dict(tickangle=-30),
    yaxis=dict(range=[0, 100]),
    margin=dict(t=40, b=40, l=40, r=40)
)

# fig.update_traces(
#     text=df_bydim["score"].round(1).astype(str),
#     textposition="outside",
#     textfont=dict(color="white")
# )

fig.write_html("www/img/interactive_by_dim.html", include_plotlyjs="cdn", full_html=False)
