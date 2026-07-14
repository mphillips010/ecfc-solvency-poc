from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import (
    calculate_contributions,
    calculate_solvency,
    category_summary,
)
from src.validation import validate_inputs
from src.charts import trend_chart, category_chart, movement_waterfall

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="ECFC Solvency Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------
# PAGE STYLING
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        .dashboard-title {
            color: #111827;
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
        }

        .dashboard-subtitle {
            color: #6B7280;
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 155px;
            border: 1px solid rgba(0, 0, 0, 0.08);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07);
        }

        .metric-label {
            font-size: 0.88rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .metric-value {
            font-size: 2.05rem;
            font-weight: 800;
            margin-top: 8px;
            margin-bottom: 5px;
        }

        .metric-detail {
            font-size: 0.88rem;
            line-height: 1.35;
        }

        .metric-status {
            display: inline-block;
            margin-top: 10px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
        }

        .card-green {
            background: #DCFCE7;
            border-left: 8px solid #16A34A;
            color: #166534;
        }

        .card-green .metric-status {
            background: #BBF7D0;
            color: #166534;
        }

        .card-amber {
            background: #FEF3C7;
            border-left: 8px solid #F59E0B;
            color: #92400E;
        }

        .card-amber .metric-status {
            background: #FDE68A;
            color: #92400E;
        }

        .card-red {
            background: #FEE2E2;
            border-left: 8px solid #DC2626;
            color: #991B1B;
        }

        .card-red .metric-status {
            background: #FECACA;
            color: #991B1B;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def card_class(status: str) -> str:
    """Return the CSS class used for the metric card."""
    normalised = str(status).upper().strip()
    if normalised == "GREEN":
        return "card-green"
    if normalised == "RED":
        return "card-red"
    return "card-amber"


def metric_card(
    label: str,
    value: str,
    detail: str,
    status: str,
) -> None:
    """Render a metric card with explicit status colouring."""
    normalised = str(status).upper().strip()
    st.markdown(
        f"""
        <div class="metric-card {card_class(normalised)}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
            <div class="metric-status">{normalised}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "line_items.csv"),
        pd.read_csv(DATA_DIR / "category_weightings.csv"),
        pd.read_csv(DATA_DIR / "thresholds.csv"),
        pd.read_csv(DATA_DIR / "movement_history.csv"),
    )


line_items, default_weights, thresholds, movement_history = load_data()

st.markdown(
    '<div class="dashboard-title">ECFC Solvency Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dashboard-subtitle">'
    '13-week liquidity and 18-month sustainability, with category assumptions, '
    'trend analysis and line-item audit trail.'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# CONTROLS
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Model controls")
    scenario = st.selectbox("Scenario", ["base"], index=0)
    entity_view = st.selectbox(
        "Entity view",
        ["Consolidated", "Club", "Trust"],
    )
    st.caption(
        "The consolidated view removes records flagged as intercompany. "
        "Standalone views retain the selected entity."
    )

if entity_view == "Club":
    filtered_items = line_items[line_items["entity"].eq("Club")].copy()
    consolidated = False
elif entity_view == "Trust":
    filtered_items = line_items[line_items["entity"].eq("Trust")].copy()
    consolidated = False
else:
    filtered_items = line_items.copy()
    consolidated = True

# ---------------------------------------------------------------------
# ASSUMPTIONS
# Store assumptions in the CSV as decimals (1.00 = 100%).
# Display and edit them here as whole-number percentages (100 = 100%).
# Do not use Streamlit's percentage formatter, because that treats 1 as 1%.
# ---------------------------------------------------------------------
display_weights = default_weights.copy()
display_weights["weight_3_month"] = (
    pd.to_numeric(display_weights["weight_3_month"], errors="coerce") * 100
)
display_weights["weight_18_month"] = (
    pd.to_numeric(display_weights["weight_18_month"], errors="coerce") * 100
)

with st.expander("Category assumptions", expanded=False):
    st.write(
        "Enter assumptions as whole-number percentages. "
        "For example, enter **100** for 100%, **75** for 75%, and **0** for 0%."
    )

    edited_display_weights = st.data_editor(
        display_weights,
        disabled=["direction", "category", "application_note"],
        column_config={
            "direction": st.column_config.TextColumn("Contributor"),
            "category": st.column_config.TextColumn("Category"),
            "weight_3_month": st.column_config.NumberColumn(
                "3-month weighting (%)",
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.0f",
                help="Whole-number percentage: 100 means 100%.",
            ),
            "weight_18_month": st.column_config.NumberColumn(
                "18-month weighting (%)",
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.0f",
                help="Whole-number percentage: 100 means 100%.",
            ),
            "application_note": st.column_config.TextColumn("Application note"),
        },
        hide_index=True,
        use_container_width=True,
        key="category_weighting_editor",
    )

# Convert whole-number percentages back to decimal values for the engine.
edited_weights = edited_display_weights.copy()
edited_weights["weight_3_month"] = (
    pd.to_numeric(edited_weights["weight_3_month"], errors="coerce") / 100
)
edited_weights["weight_18_month"] = (
    pd.to_numeric(edited_weights["weight_18_month"], errors="coerce") / 100
)

threshold_map = thresholds.set_index("metric")
short_min = float(threshold_map.loc["3_month", "minimum_threshold"])
short_target = float(threshold_map.loc["3_month", "target_threshold"])
long_min = float(threshold_map.loc["18_month", "minimum_threshold"])
long_target = float(threshold_map.loc["18_month", "target_threshold"])

short_detail = calculate_contributions(
    filtered_items,
    edited_weights,
    "3_month",
    scenario=scenario,
    consolidated=consolidated,
)
long_detail = calculate_contributions(
    filtered_items,
    edited_weights,
    "18_month",
    scenario=scenario,
    consolidated=consolidated,
)

short_result = calculate_solvency(
    short_detail,
    "3_month",
    short_min,
    short_target,
)
long_result = calculate_solvency(
    long_detail,
    "18_month",
    long_min,
    long_target,
)

# ---------------------------------------------------------------------
# HOMEPAGE
# ---------------------------------------------------------------------
st.subheader("Headline solvency outputs")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "3-month solvency ratio",
        f"{short_result.ratio:.2f}x",
        f"Minimum {short_min:.2f}x · Target {short_target:.2f}x",
        short_result.status,
    )

with c2:
    metric_card(
        "3-month headroom",
        f"£{short_result.headroom:,.0f}",
        (
            f"Positive resources £{short_result.positives:,.0f}<br>"
            f"Requirements £{short_result.negatives:,.0f}"
        ),
        short_result.status,
    )

with c3:
    metric_card(
        "18-month solvency ratio",
        f"{long_result.ratio:.2f}x",
        f"Minimum {long_min:.2f}x · Target {long_target:.2f}x",
        long_result.status,
    )

with c4:
    metric_card(
        "18-month headroom",
        f"£{long_result.headroom:,.0f}",
        (
            f"Positive resources £{long_result.positives:,.0f}<br>"
            f"Requirements £{long_result.negatives:,.0f}"
        ),
        long_result.status,
    )

st.write("")

left, right = st.columns(2)
with left:
    st.plotly_chart(
        trend_chart(
            movement_history,
            "3_month",
            short_min,
            short_target,
        ),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        trend_chart(
            movement_history,
            "18_month",
            long_min,
            long_target,
        ),
        use_container_width=True,
    )

short_summary = category_summary(short_detail)
long_summary = category_summary(long_detail)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        category_chart(
            short_summary,
            "3-month contribution by category",
        ),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        category_chart(
            long_summary,
            "18-month contribution by category",
        ),
        use_container_width=True,
    )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Movements",
        "Contribution detail",
        "Line-item calculation",
        "Validation",
    ]
)

with tab1:
    m1, m2 = st.columns(2)
    with m1:
        st.plotly_chart(
            movement_waterfall(movement_history, "3_month"),
            use_container_width=True,
        )
    with m2:
        st.plotly_chart(
            movement_waterfall(movement_history, "18_month"),
            use_container_width=True,
        )

with tab2:
    p1, p2 = st.columns(2)
    with p1:
        st.dataframe(
            short_summary,
            hide_index=True,
            use_container_width=True,
        )
    with p2:
        st.dataframe(
            long_summary,
            hide_index=True,
            use_container_width=True,
        )

with tab3:
    horizon = st.radio(
        "Calculation horizon",
        ["3_month", "18_month"],
        horizontal=True,
    )
    detail = short_detail if horizon == "3_month" else long_detail

    display_columns = [
        "record_id",
        "entity",
        "source_statement",
        "source_account",
        "line_item",
        "forecast_period",
        "due_date",
        "direction",
        "category",
        "gross_amount",
        "applied_relevance",
        "applied_weight",
        "probability",
        "liquidity_haircut",
        "discount_factor",
        "eligible_contribution",
        "signed_contribution",
        "data_quality_status",
    ]

    st.dataframe(
        detail[display_columns].sort_values(
            ["direction", "category", "eligible_contribution"],
            ascending=[True, True, False],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download calculation detail",
        detail[display_columns].to_csv(index=False),
        file_name=f"solvency_calculation_{horizon}.csv",
        mime="text/csv",
    )

with tab4:
    checks = validate_inputs(filtered_items, edited_weights)
    st.dataframe(
        checks,
        hide_index=True,
        use_container_width=True,
    )

    failed = checks["status"].eq("FAIL").sum()
    if failed:
        st.error(f"{failed} validation check(s) failed.")
    else:
        st.success("All proof-of-concept validation checks passed.")
