
import pandas as pd
import plotly.express as px
from workspace.utils.helpers import model_name, dts_map, model_colors
from workspace.eval import build_eval_df
from task_spec import task_specs

# Define domain groupings
domain_map = {
    "BRFSS": "Health", "NHANES": "Health", "NSDUH": "Health", "MEPS": "Health",
    "GSS": "Social", "FBI Arrests": "Social", "IPEDS": "Social",
    "BLS": "Economic", "ACS": "Economic", "SCF": "Economic"
}

domain_order = ["Health", "Social", "Economic"]

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]
eval_df, eval_map = build_eval_df(models, task_specs)

# Aggregate
df = eval_df.groupby(["model", "dataset"]).agg(score=("score", "mean")).reset_index()
df["dataset"] = df["dataset"].map(dts_map)
df["Model"] = df["model"].apply(model_name)

# Assign domains
domain_map = {
    "BRFSS": "Health", "NHANES": "Health", "NSDUH": "Health", "MEPS": "Health",
    "GSS": "Social", "FBI Arrests": "Social", "IPEDS": "Social",
    "BLS": "Economic", "ACS": "Economic", "SCF": "Economic"
}
domain_order = ["Health", "Social", "Economic"]
df["Domain"] = df["dataset"].map(domain_map)
df["Domain"] = pd.Categorical(df["Domain"], categories=domain_order, ordered=True)

# X-axis: Model | Dataset
df["x"] = df["Model"] + " | " + df["dataset"]

# ✔️ CORRECT x-axis order: domain → model → dataset_in_that_domain
x_order = []
for domain in domain_order:
    domain_datasets = sorted([d for d, dom in domain_map.items() if dom == domain])
    for model in sorted(df["Model"].unique()):
        for dataset in domain_datasets:
            # Only add if model-dataset exists in df
            if ((df["Model"] == model) & (df["dataset"] == dataset)).any():
                x_order.append(f"{model} | {dataset}")

df["x"] = pd.Categorical(df["x"], categories=x_order, ordered=True)

# Plot
fig = px.bar(
    df, x="x", y="score", color="Model",
    text_auto=".0f", color_discrete_map=model_colors
)

fig.update_layout(
    title="Scores by Domain",
    xaxis_title="",
    yaxis_title="Average Score",
    yaxis=dict(range=[0, 100]),
    bargap=0.05,
    plot_bgcolor="#0d1b2a",
    paper_bgcolor="#0d1b2a",
    font=dict(color="white"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_tickangle=45,
    xaxis_type='category'
)

fig.update_xaxes(categoryorder='array', categoryarray=x_order)
fig.update_traces(marker_line_color="black", marker_line_width=0.5)

# Vertical lines between domains
domain_boundaries = []
cur = 0
for dom in domain_order:
    n = df[df["Domain"] == dom].shape[0]
    domain_boundaries.append((cur, cur + n))
    cur += n

for start, end in domain_boundaries[1:]:
    fig.add_vline(
        x=start - 0.5,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )

# Domain labels in yellow above each group
for (start, end), dom in zip(domain_boundaries, domain_order):
    fig.add_annotation(
        x=(start + end - 1) / 2,
        y=87,
        text=dom,
        showarrow=False,
        font=dict(color="#d4af37", size=32)
    )

# Clean x-axis labels (drop model name)
fig.update_xaxes(tickvals=x_order, ticktext=[x.split(" | ")[1] for x in x_order])

fig.write_html("www/img/interactive_ld_bydomain.html", include_plotlyjs="cdn")
