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
    page_title="ECFC Solvency POC",
    page_icon="⚽",
    layout="wide",
)

st.title("ECFC Solvency Calculation and Reporting Dashboard")
st.caption(
    "Proof of concept: 13-week liquidity and 18-month sustainability, "
    "with category assumptions, item-level audit trail, trends and movements."
)


@st.cache_data
def load_data():
    return (
        pd.read_csv(DATA_DIR / "line_items.csv"),
        pd.read_csv(DATA_DIR / "category_weightings.csv"),
        pd.read_csv(DATA_DIR / "thresholds.csv"),
        pd.read_csv(DATA_DIR / "movement_history.csv"),
    )


line_items, default_weights, thresholds, movement_history = load_data()

with st.sidebar:
    st.header("Model controls")
    scenario = st.selectbox(
        "Scenario",
        ["base"],
        index=0,
    )
    entity_view = st.selectbox(
        "Entity view",
        ["Consolidated", "Club", "Trust"],
    )
    st.caption(
        "Consolidated view eliminates items flagged as intercompany. "
        "Standalone views retain the relevant entity data."
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

st.subheader("Category assumptions")
st.write(
    "Edit the category-level percentages below. All mapped line items update automatically."
)
edited_weights = st.data_editor(
    default_weights,
    disabled=["direction", "category", "application_note"],
    column_config={
        "weight_3_month": st.column_config.NumberColumn(
            "3-month weighting", min_value=0.0, max_value=1.0, step=0.05, format="%.0f%%"
        ),
        "weight_18_month": st.column_config.NumberColumn(
            "18-month weighting", min_value=0.0, max_value=1.0, step=0.05, format="%.0f%%"
        ),
    },
    hide_index=True,
    use_container_width=True,
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
    short_detail, "3_month", short_min, short_target
)
long_result = calculate_solvency(
    long_detail, "18_month", long_min, long_target
)

st.divider()
st.subheader("Headline solvency outputs")

c1, c2, c3, c4 = st.columns(4)
c1.metric("3-month solvency ratio", f"{short_result.ratio:.2f}x")
c2.metric("3-month headroom", f"£{short_result.headroom:,.0f}", short_result.status)
c3.metric("18-month solvency ratio", f"{long_result.ratio:.2f}x")
c4.metric("18-month headroom", f"£{long_result.headroom:,.0f}", long_result.status)

status_cols = st.columns(2)
with status_cols[0]:
    st.info(
        f"3-month: positive resources £{short_result.positives:,.0f}; "
        f"negative requirements £{short_result.negatives:,.0f}; "
        f"minimum {short_result.minimum_threshold:.2f}x; "
        f"target {short_result.target_threshold:.2f}x."
    )
with status_cols[1]:
    st.info(
        f"18-month: positive resources £{long_result.positives:,.0f}; "
        f"negative requirements £{long_result.negatives:,.0f}; "
        f"minimum {long_result.minimum_threshold:.2f}x; "
        f"target {long_result.target_threshold:.2f}x."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Trend", "Movements", "Contributions", "Line-item calculation", "Validation"]
)

with tab1:
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(
            trend_chart(movement_history, "3_month", short_min, short_target),
            use_container_width=True,
        )
    with t2:
        st.plotly_chart(
            trend_chart(movement_history, "18_month", long_min, long_target),
            use_container_width=True,
        )

with tab2:
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

with tab3:
    p1, p2 = st.columns(2)
    with p1:
        short_summary = category_summary(short_detail)
        st.plotly_chart(
            category_chart(short_summary, "3-month contribution by category"),
            use_container_width=True,
        )
        st.dataframe(short_summary, hide_index=True, use_container_width=True)
    with p2:
        long_summary = category_summary(long_detail)
        st.plotly_chart(
            category_chart(long_summary, "18-month contribution by category"),
            use_container_width=True,
        )
        st.dataframe(long_summary, hide_index=True, use_container_width=True)

with tab4:
    horizon = st.radio("Calculation horizon", ["3_month", "18_month"], horizontal=True)
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

with tab5:
    checks = validate_inputs(filtered_items, edited_weights)
    st.dataframe(checks, hide_index=True, use_container_width=True)
    failed = checks["status"].eq("FAIL").sum()
    if failed:
        st.error(f"{failed} validation check(s) failed.")
    else:
        st.success("All proof-of-concept validation checks passed.")
