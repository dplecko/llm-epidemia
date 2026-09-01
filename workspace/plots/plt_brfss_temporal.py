"""Deviation strip plot: BRFSS benchmark score minus each series' own
across-year mean, one jittered column per survey year, faceted by
Low-Dimensional (by-state) vs. High-Dimensional (income-conditioned tasks
excluded) setting. Run from the repo root:
    python workspace/plots/plt_brfss_temporal.py
"""
import pandas as pd
from plotnine import *

YEAR_COLORS = {2015: "#9ec5f4", 2018: "#5598e7", 2020: "#256abf", 2023: "#104281"}

df = pd.read_csv("data/benchmark/brfss_temporal_scores.csv")
df["series"] = df["outcome"] + "_" + df["cond_vars"]
df["Setting"] = df["setting"].map({"low": "Low-Dimensional", "high": "High-Dimensional"})
df = df[df["available"]].copy()
df["Setting"] = pd.Categorical(df["Setting"], categories=["Low-Dimensional", "High-Dimensional"], ordered=True)
df["year"] = pd.Categorical(df["year"], sorted(YEAR_COLORS))
df["delta"] = df["score"] - df.groupby(["Setting", "series", "model"])["score"].transform("mean")

stats = df.groupby(["Setting", "year"], observed=True)["delta"].agg(
    q1=lambda s: s.quantile(0.25), med="median", q3=lambda s: s.quantile(0.75)).reset_index()

plt_brfss_temporal = (
    ggplot(df, aes("year", "delta", color="year"))
    + geom_hline(yintercept=0, linetype="dashed", color="grey")
    + geom_jitter(width=0.15, size=2.2, alpha=0.55, show_legend=False)
    + geom_crossbar(stats, aes("year", "med", ymin="q1", ymax="q3"), width=0.35,
                     color="black", fill="white", alpha=0, size=0.6, inherit_aes=False)
    + scale_color_manual(values=YEAR_COLORS)
    + labs(x="BRFSS survey year", y="Score deviation from own 4-year mean")
    + theme_bw()
    + theme(panel_background=element_rect(fill="white"), plot_background=element_rect(fill="white"))
    + facet_wrap("~ Setting")
)

plt_brfss_temporal
plt_brfss_temporal.save("data/plots/brfss_temporal_deviation.png", dpi=300, width=10, height=5)
