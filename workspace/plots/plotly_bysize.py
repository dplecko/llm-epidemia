
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from utils.helpers import model_name, model_colors
from eval import build_eval_df
from task_spec import task_specs_hd

# models and data
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
sizes = {
    "phi4": 1.8,
    "mistral_7b_instruct": 7,
    "deepseek_7b_chat": 7,
    "llama3_8b_instruct": 8,
    "gemma3_27b_instruct": 27,
    "llama3_70b_instruct": 70,
    "gpt-4.1": 1800
}

prob = os.getenv("PROB_EVAL", "false").lower() == "true"
if prob:
    models = models + ["gpt-4.1"]
eval_hd, _ = build_eval_df(models, task_specs_hd, prob=prob)

# assume eval_hd already defined as above
df_size = eval_hd.groupby("model").agg(score=("score", "mean")).reset_index()
df_size["size"] = df_size["model"].map(sizes)
df_size["Model"] = df_size["model"].apply(model_name)

fig = px.scatter(
    df_size,
    x="size",
    y="score",
    text="Model",
    size="size",  # ✅ size proportional to model size
    size_max=40,  # optional: cap max size
    labels={"size": "Model Size (B Params)", "score": "Average Score"},
)

fig.update_traces(
    marker=dict(size=12, color=[model_colors[m] for m in df_size["Model"]]),
    textposition="top center"
)

fig.add_shape(
    type="line", x0=min(df_size["size"]), x1=max(df_size["size"]),
    y0=100, y1=100,
    line=dict(color="lime", dash="dash")
)

fig.update_layout(
    title="Performance by Model Size",
    plot_bgcolor="#0d1b2a",
    paper_bgcolor="#0d1b2a",
    font=dict(color="white"),
    yaxis=dict(range=[0, 105]),
    margin=dict(t=40, b=40)
)

# Optional: log x-axis
if prob:
    fig.update_xaxes(type="log")

file_name = "interactive_bysize.html"
if prob:
    file_name = "PROB_" + file_name
fig.write_html(os.path.join("www/img", file_name), include_plotlyjs="cdn", full_html=False)
