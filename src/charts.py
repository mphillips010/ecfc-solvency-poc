from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def trend_chart(
    history: pd.DataFrame,
    horizon: str,
    minimum: float,
    target: float,
    title: str = None,
) -> go.Figure:
    """Generate formal trend chart with threshold indicators."""
    df = history[history["horizon"].eq(horizon)].copy()
    df["reporting_date"] = pd.to_datetime(df["reporting_date"])
    ratio = (
        df.groupby("reporting_date", as_index=False)
        .agg(positives=("positive_resources", "sum"), negatives=("negative_requirements", "sum"))
    )
    ratio["ratio"] = ratio["positives"] / ratio["negatives"]

    horizon_label = "13-Week" if horizon == "3_month" else "18-Month"
    default_title = f"{horizon_label} Solvency Ratio — Historical Trend"
    
    fig = px.line(ratio, x="reporting_date", y="ratio", markers=True)
    
    fig.add_hline(
        y=minimum,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Minimum ({minimum:.2f}x)",
        annotation_position="right",
    )
    fig.add_hline(
        y=target,
        line_dash="dot",
        line_color="#059669",
        annotation_text=f"Target ({target:.2f}x)",
        annotation_position="right",
    )
    
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Reporting Date",
        yaxis_title="Coverage Ratio (x)",
        hovermode="x unified",
        plot_bgcolor="rgba(249, 250, 251, 1)",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11, color="#374151"),
        title_font=dict(size=13, color="#0f172a", family="Arial, sans-serif"),
        showlegend=False,
        margin=dict(l=70, r=150, t=60, b=60),
    )
    
    fig.update_traces(
        line=dict(color="#0f172a", width=2),
        marker=dict(size=6, symbol="circle"),
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#e5e7eb",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#e5e7eb",
        zeroline=False,
    )
    
    return fig


def category_chart(summary: pd.DataFrame, title: str) -> go.Figure:
    """Generate formal category contribution chart."""
    df = summary.copy()
    df["signed"] = df["eligible_contribution"].where(
        df["direction"].eq("positive"), -df["eligible_contribution"]
    )
    
    # Map direction to formal labels
    df["contributor_type"] = df["direction"].map({
        "positive": "Resources",
        "negative": "Requirements",
    })
    
    fig = px.bar(
        df,
        x="category",
        y="signed",
        color="contributor_type",
        barmode="relative",
        title=title,
        color_discrete_map={
            "Resources": "#059669",
            "Requirements": "#dc2626",
        },
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Eligible Contribution (£)",
        hovermode="x unified",
        plot_bgcolor="rgba(249, 250, 251, 1)",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11, color="#374151"),
        title_font=dict(size=13, color="#0f172a", family="Arial, sans-serif"),
        legend=dict(
            title="Contributor Type",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        margin=dict(l=70, r=100, t=60, b=60),
    )
    
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#e5e7eb",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#d1d5db",
    )
    
    return fig


def movement_waterfall(history: pd.DataFrame, horizon: str) -> go.Figure:
    """Generate formal movement waterfall chart."""
    df = history[history["horizon"].eq(horizon)].copy()
    df["reporting_date"] = pd.to_datetime(df["reporting_date"])
    dates = sorted(df["reporting_date"].unique())
    
    if len(dates) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for movement analysis", showarrow=False)
        return fig

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

    horizon_label = "13-Week" if horizon == "3_month" else "18-Month"
    
    fig = go.Figure(
        go.Waterfall(
            measure=["absolute"] + ["relative"] * len(all_categories) + ["total"],
            x=["Opening"] + all_categories + ["Closing"],
            y=[opening] + changes + [closing],
            connector={"line": {"color": "#374151"}},
            decreasing={"marker": {"color": "#dc2626"}},
            increasing={"marker": {"color": "#059669"}},
            totals={"marker": {"color": "#0f172a"}},
        )
    )
    
    fig.update_layout(
        title=f"{horizon_label} Headroom — Period-on-Period Movement",
        yaxis_title="Headroom (£)",
        xaxis_title="Movement Category",
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor="rgba(249, 250, 251, 1)",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11, color="#374151"),
        title_font=dict(size=13, color="#0f172a", family="Arial, sans-serif"),
        margin=dict(l=70, r=100, t=60, b=60),
    )
    
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#e5e7eb",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#d1d5db",
    )
    
    return fig
