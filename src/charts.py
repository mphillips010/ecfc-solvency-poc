from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def trend_chart(history: pd.DataFrame, horizon: str, minimum: float, target: float):
    df = history[history["horizon"].eq(horizon)].copy()
    df["reporting_date"] = pd.to_datetime(df["reporting_date"])
    ratio = (
        df.groupby("reporting_date", as_index=False)
        .agg(positives=("positive_resources", "sum"), negatives=("negative_requirements", "sum"))
    )
    ratio["ratio"] = ratio["positives"] / ratio["negatives"]

    fig = px.line(ratio, x="reporting_date", y="ratio", markers=True)
    fig.add_hline(y=minimum, line_dash="dash", annotation_text="Minimum")
    fig.add_hline(y=target, line_dash="dot", annotation_text="Target")
    fig.update_layout(
        title=f"{'3-month' if horizon == '3_month' else '18-month'} solvency ratio trend",
        xaxis_title="Reporting date",
        yaxis_title="Coverage ratio",
        legend_title_text="",
    )
    return fig


def category_chart(summary: pd.DataFrame, title: str):
    df = summary.copy()
    df["signed"] = df["eligible_contribution"].where(
        df["direction"].eq("positive"), -df["eligible_contribution"]
    )
    fig = px.bar(
        df,
        x="category",
        y="signed",
        color="direction",
        barmode="relative",
        title=title,
    )
    fig.update_layout(xaxis_title="", yaxis_title="Eligible contribution (£)")
    return fig


def movement_waterfall(history: pd.DataFrame, horizon: str):
    df = history[history["horizon"].eq(horizon)].copy()
    df["reporting_date"] = pd.to_datetime(df["reporting_date"])
    dates = sorted(df["reporting_date"].unique())
    if len(dates) < 2:
        return go.Figure()

    current, previous = dates[-1], dates[-2]
    prev = df[df["reporting_date"].eq(previous)].set_index("movement_category")
    curr = df[df["reporting_date"].eq(current)].set_index("movement_category")

    all_categories = sorted(set(prev.index).union(curr.index))
    changes = []
    for category in all_categories:
        prior = float(prev.loc[category, "net_headroom"]) if category in prev.index else 0.0
        latest = float(curr.loc[category, "net_headroom"]) if category in curr.index else 0.0
        changes.append(latest - prior)

    opening = float(prev["net_headroom"].sum())
    closing = float(curr["net_headroom"].sum())

    fig = go.Figure(
        go.Waterfall(
            measure=["absolute"] + ["relative"] * len(all_categories) + ["total"],
            x=["Opening"] + all_categories + ["Closing"],
            y=[opening] + changes + [closing],
            connector={"line": {"color": "rgb(63,63,63)"}},
        )
    )
    fig.update_layout(
        title=f"Movement in {'3-month' if horizon == '3_month' else '18-month'} headroom",
        yaxis_title="Headroom (£)",
        showlegend=False,
    )
    return fig
