
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from workspace.utils.helpers import model_name
from workspace.eval import build_eval_df
from workspace.task_spec import task_specs, task_specs_hd

# Dummy data — replace with your actual df_scores
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_df, _ = build_eval_df(models, task_specs + task_specs_hd, prob = False)

# Process
agg = eval_df.groupby("model")[["score"]].mean().round().astype(int).reset_index()
agg["model_display"] = agg["model"].apply(model_name)
agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

# Colors
colors = qualitative.Dark24
color_map = {m: colors[i % len(colors)] for i, m in enumerate(agg["model_display"])}

# Plot
fig = go.Figure(go.Bar(
    x=agg["model_display"],
    y=agg["score"],
    text=agg["score"].astype(int),
    textposition='outside',
    marker=dict(color=[color_map[m] for m in agg["model_display"]])
))
fig.update_traces(textfont_color="white", cliponaxis=False)

perfect_score = 100  # adjust to your real perfect total

fig.update_layout(
    font=dict(size=14, color="#f8f9fa", family="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"),
    plot_bgcolor="#0d1b2a",
    paper_bgcolor="#0d1b2a",
    yaxis=dict(
        title="Score",
        range=[0, perfect_score],
        showgrid=True,
        gridcolor="#2e3b55",
        color="#f8f9fa"
    ),
    xaxis=dict(
        title="Model",
        tickangle=-0,
        color="#f8f9fa"
    ),
    title_font=dict(color="#e0e1dd"),
    margin=dict(t=40, b=30),
    height=400
)

fig.add_shape(
    type="line",
    x0=-0.5, x1=len(agg) - 0.5,
    y0=perfect_score, y1=perfect_score,
    line=dict(color="lime", dash="dash")
)
fig.add_annotation(
    x=0.5, y=perfect_score - 5,
    text="Perfect Score",
    showarrow=False,
    font=dict(color="lime")
)

fig.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="#0d1b2a",  # restore your dark background
    plot_bgcolor="#0d1b2a"
)

# Export
fig.write_html("www/img/distribution.html", include_plotlyjs="cdn", full_html=True)

