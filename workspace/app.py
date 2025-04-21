from shiny import App, render, reactive
from shiny import ui
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import sys
import os
import pdb

sys.path.append(os.path.abspath("workspace"))

from helpers import model_name, model_unname
from build_eval_df import build_eval_df
from task_spec import task_specs
from diagnostic_plots import make_detail_plot, make_detail_plotnine

# ---- Load Data ----
raw_models = ["llama3_8b_instruct", "mistral_7b_instruct", "phi4",
              "gemma3_27b_instruct", "llama3_70b_instruct"]
tasks = task_specs
df_scores = build_eval_df(raw_models, tasks)
df_scores["model_display"] = model_name(df_scores["model"])

def model_bar_plot():
    # Score aggregation
    agg = df_scores.groupby("model")[["score"]].sum().reset_index()
    # print(df_scores.groupby("model")["score"].sum())  # DEBUG

    agg["model_display"] = model_name(agg["model"])
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

    # Model-specific colors
    unique_models = agg["model_display"].unique()
    colors = qualitative.Dark24
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(unique_models)}

    fig = go.Figure(go.Bar(
        x=agg["model_display"],
        y=agg["score"].astype(float).tolist(),
        text=agg["score"].astype(int).tolist(),
        textposition='outside',
        marker=dict(color=colors)
    ))
    fig.update_traces(textfont_color="white", cliponaxis=False)

    # Calculate perfect score
    perfect_score = 100 * len(tasks)

    fig.update_layout(
        title="Total Benchmark Score by Model",
        font=dict(size=14, color="white"),
        plot_bgcolor="#222222",
        paper_bgcolor="#222222",
        yaxis=dict(
            title="Score",
            range=[0, perfect_score],  # ✅ fix for visual scale
            showgrid=True,
            gridcolor="#444444",
            color="white"
        ),
        xaxis=dict(
            title="Model",
            tickangle=-15,
            color="white"
        ),
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
        x=0.5, y=perfect_score - 20,
        text="Perfect Score",
        showarrow=False,
        font=dict(color="lime")
    )

    return fig


# ---- UI ----
app_ui = ui.page_fluid(
    ui.tags.link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/darkly/bootstrap.min.css"
    ),
    ui.tags.script(src="https://cdn.plot.ly/plotly-2.27.0.min.js"),
    ui.h2("L1 Faithfulness Benchmark", style="text-align: center;"),
    ui.layout_columns(
        ui.column(8, ui.output_ui("bar"), offset=2)
    ),
    ui.br(),
    ui.layout_columns(
        ui.column(10,
            ui.output_ui("tbl"), offset=2        ),
        ui.column(10,
            ui.output_ui("detail_plot")
        )
    )
)

# ---- Server ----
def server(input, output, session):

    @output
    @render.ui
    def bar():
        return ui.HTML(model_bar_plot().to_html(full_html=False, include_plotlyjs=False))

    @output
    @render.ui
    def tbl():
        # Fresh pivoted table (tasks as rows, models as columns)
        df = df_scores[["task_id", "model", "score"]].copy()
        df["model_display"] = model_name(df["model"])

        df_wide = df.pivot(index="task_id", columns="model_display", values="score").reset_index()
        model_cols = [col for col in df_wide.columns if col != "task_id"]
        df_wide["task_name"] = [x["name"] for x in tasks]

        table = "<table style='border-collapse: collapse; width: 100%; text-align: center;'>"
        table += "<thead><tr><th style='border: 1px solid #ccc;'>Task</th>"
        for model in model_cols:
            table += f"<th style='border: 1px solid #ccc;'>{model}</th>"
        table += "</tr></thead><tbody>"

        for i, row in df_wide.iterrows():
            table += f"<tr><td style='border: 1px solid #ccc;'>{row['task_name']}</td>"
            for model in model_cols:
                val = row[model]
                val_str = "NA" if pd.isna(val) else f"{int(val)}"
                table += f"<td style='border: 1px solid #ccc;' onclick=\"sendSelection('{model}', {row['task_id']})\">{val_str}</td>"
            table += "</tr>"

        table += "</tbody></table>"

        script = """
        <script>
        function sendSelection(model, task_idx) {
            Shiny.setInputValue("sel_model", model, {priority: "event"});
            Shiny.setInputValue("task_idx", task_idx, {priority: "event"});
        }
        </script>
        """

        return ui.HTML(table + script)

    @output
    @render.ui
    def detail_plot():

        if input.sel_model() is None or input.task_idx() is None:
            return ui.h5("Click a score cell to view the breakdown.")
        
        model_raw = model_unname(input.sel_model())
        task_idx = input.task_idx()
        task = tasks[task_idx]

        try:
            row = df_scores[
                (df_scores["model"] == model_raw) & (df_scores["task_id"] == task_idx)
            ].iloc[0]
            df = row["eval"]
        except IndexError:
            return ui.h5("No evaluation data found.")

        html = make_detail_plotnine(df, task)
        return ui.HTML(html)

app = App(app_ui, server)