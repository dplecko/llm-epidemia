
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pdb
from plotnine import *
from io import BytesIO
import base64
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate

def make_detail_plot(df, task):
    if df.shape[0] == 1:
        # Marginal: density-like histogram
        true_vals = df["true_vals"].iloc[0]["vals"]
        true_wgh = df["true_vals"].iloc[0]["wgh"]
        mod_vals = df["mod_vals"].iloc[0]["vals"]

        plt_dat = pd.DataFrame({
            "vals": true_vals + mod_vals,
            "weight": true_wgh + [1] * len(mod_vals),
            "type": ["Reality"] * len(true_vals) + ["Model"] * len(mod_vals)
        })

        fig = px.histogram(
            plt_dat,
            x="vals",
            color="type",
            nbins=50,
            histnorm="probability density",
            opacity=0.6,
            barmode="overlay",
        )
        fig.update_layout(title=task["name"])

    elif "p_true" in df.columns and "p_mod" in df.columns:
        # Binary or continuous
        fig = px.scatter(df, x="p_true", y="p_mod", text="cond", title=task["name"])
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="orange", dash="dash"))
        fig.update_layout(
            xaxis_title="True Mean", yaxis_title="Model's Mean"
        )

        if task.get("levels") and len(task["levels"]) == 2:
            fig.update_layout(
                xaxis=dict(tickformat=".0%", range=[0, 1]),
                yaxis=dict(tickformat=".0%", range=[0, 1])
            )

    elif hasattr(df, "attrs") and "distr" in df.attrs:
        distr = df.attrs["distr"]
        # pdb.set_trace()  # DEBUG
        distr["prop"] = distr["prop"].apply(float)
        fig = px.bar(
            distr,
            x="lvl_names",
            y="prop",
            color="type",
            facet_col="cond",
            barmode="group",
            category_orders={"lvl": sorted(set(distr["lvl"]))},
            title=task["name"],
        )
        fig.for_each_xaxis(lambda ax: ax.update(title=None))
        fig.update_layout(
            yaxis_title="Proportion",
            annotations=[
                dict(
                    text="Level",  # shared label
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14),
                    xanchor="center"
                )
            ]
        )

    else:
        fig = px.line(x=[0, 1, 2], y=[2, 4, 1], title="No usable eval data")

    return fig

def dark_theme():
    return theme(
        panel_background=element_rect(fill="#222222"),
        plot_background=element_rect(fill="#222222"),
        panel_grid_major=element_line(color="#444444"),
        panel_grid_minor=element_line(color="#333333"),
        text=element_text(color="white"),
        axis_text=element_text(color="white"),
        axis_title=element_text(color="white"),
        legend_background=element_rect(fill="#222222"),
        legend_key=element_rect(fill="#222222"),
        legend_text=element_text(color="white"),
        strip_background=element_rect(fill="#333333"),
        strip_text=element_text(color="white")
    )

def kde_density(vals, wgh, label):
    kde = KDEUnivariate(vals)
    kde.fit(weights=wgh, bw="scott", fft=False)
    return pd.DataFrame({"x": kde.support, "y": kde.density, "type": label})

def make_detail_plotnine(df, task, single_cond=None):
    if df.shape[0] == 1:
        # Marginal: histogram via geom_density
        true_vals = np.array(df["true_vals"].iloc[0]["vals"], dtype=float).copy()
        true_wgh = np.array(df["true_vals"].iloc[0]["wgh"], dtype=float).copy()
        mod_vals = np.array(df["mod_vals"].iloc[0]["vals"], dtype=float).copy()

        # Force writable:
        true_vals.setflags(write=True)
        true_wgh.setflags(write=True)
        mod_vals.setflags(write=True)

        df_real = kde_density(true_vals, true_wgh, "Reality")
        df_model = kde_density(mod_vals, np.ones_like(mod_vals), "Model")
        plot_df = pd.concat([df_real, df_model])

        p = (
            ggplot(plot_df, aes(x="x", y="y", fill="type"))
            + geom_area(alpha = 0.6, stat = "identity")
            + labs(title=task["name"], x="Value", y="Density")
            + scale_fill_manual(values={"Reality": "#1f77b4", "Model": "#ff7f0e"})
            + theme_bw()
            + dark_theme()
        )

    elif "p_true" in df.columns and "p_mod" in df.columns:
        # Binary or continuous
        p = (
            ggplot(df, aes(x="p_true", y="p_mod", label="cond"))
            + geom_abline(slope=1, intercept=0, linetype="dashed", color="orange")
            + geom_smooth(method="loess", color = "blue", se=True)
            + geom_point(color = "white")
            + theme_bw()
            + labs(title=task["name"], x="True Mean", y="Model's Mean")
            + dark_theme()
        )

        if task.get("levels") and len(task["levels"]) == 2:
            p += scale_x_continuous(labels=lambda l: [f"{v:.0%}" for v in l])
            p += scale_y_continuous(labels=lambda l: [f"{v:.0%}" for v in l])
            p += coord_cartesian(xlim=(0, 1), ylim=(0, 1))

    elif hasattr(df, "attrs") and "distr" in df.attrs:
        distr = df.attrs["distr"].copy()
        distr["prop"] = distr["prop"].astype(float)

        if single_cond is not None:
            distr = distr[distr["cond"] == single_cond]

        p = (
            ggplot(distr, aes(x="lvl_names", y="prop", fill="type"))
            + geom_col(position="dodge", color="black")
            + labs(title=f'{task["name"]} – {single_cond}', x="Level", y="Proportion")
            + scale_y_continuous(labels=lambda l: [f"{v:.0%}" for v in l])
            + theme(axis_text_x=element_text(rotation=45, ha="right"))
            + dark_theme()
        )
    else:
        return "<p>No usable eval data</p>"

    # Convert plot to HTML image
    return p
    # buf = BytesIO()
    # p.save(buf, format='png', dpi=300, width=7, height=4, units="in", verbose=False)
    # img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    # return f"<div style='text-align: center;'><img src='data:image/png;base64,{img_str}' style='width: 80%;'/></div>"