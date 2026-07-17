from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np

from src.model import (
    calculate_contributions,
    calculate_solvency,
    category_summary,
)
from src.validation import validate_inputs
from src.charts import trend_chart, category_chart, movement_waterfall

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Solvency Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------
# PAGE STYLING - CLEAN AND FORMAL
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        .dashboard-header {
            border-bottom: 1px solid #d1d5db;
            padding-bottom: 2rem;
            margin-bottom: 2rem;
        }

        .dashboard-title {
            color: #111827;
            font-size: 2.25rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            line-height: 1.2;
        }

        .dashboard-subtitle {
            color: #4b5563;
            font-size: 0.95rem;
            font-weight: 400;
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }

        .disclaimer-text {
            color: #6b7280;
            font-size: 0.85rem;
            margin-top: 1rem;
            padding: 0.75rem;
            background-color: #f9fafb;
            border-left: 3px solid #d97706;
            border-radius: 4px;
        }

        .metric-card {
            border-radius: 6px;
            padding: 1.25rem;
            min-height: 160px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            background: #ffffff;
            display: flex;
            flex-direction: column;
        }

        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.875rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.5rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        }

        .metric-detail {
            font-size: 0.8rem;
            line-height: 1.5;
            color: #6b7280;
            margin-bottom: auto;
        }

        .metric-status {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.35rem 0.65rem;
            border-radius: 3px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            width: fit-content;
        }

        .card-compliant {
            background: #f0fdf4;
            border-left: 4px solid #059669;
            color: #065f46;
        }

        .card-compliant .metric-status {
            background: #d1fae5;
            color: #065f46;
        }

        .card-caution {
            background: #fffbeb;
            border-left: 4px solid #d97706;
            color: #78350f;
        }

        .card-caution .metric-status {
            background: #fef3c7;
            color: #78350f;
        }

        .card-critical {
            background: #fef2f2;
            border-left: 4px solid #dc2626;
            color: #7f1d1d;
        }

        .card-critical .metric-status {
            background: #fee2e2;
            color: #7f1d1d;
        }

        .section-header {
            color: #111827;
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 2.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 0.75rem;
        }

        .control-label {
            color: #111827;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.5rem;
        }

        .info-text {
            color: #4b5563;
            line-height: 1.6;
            font-size: 0.9rem;
        }

        .tab-description {
            color: #4b5563;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            font-weight: 500;
        }

        .sensitivity-container {
            background: #f9fafb;
            border-radius: 6px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e5e7eb;
        }

        .sensitivity-label {
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .sensitivity-value {
            color: #059669;
            font-weight: 600;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def card_class(status: str) -> str:
    """Return the CSS class used for the metric card."""
    normalised = str(status).upper().strip()
    if normalised == "GREEN":
        return "card-compliant"
    if normalised == "RED":
        return "card-critical"
    return "card-caution"


def metric_status_label(status: str) -> str:
    """Return the formal label for the metric status."""
    normalised = str(status).upper().strip()
    if normalised == "GREEN":
        return "COMPLIANT"
    if normalised == "RED":
        return "NON-COMPLIANT"
    return "AT RISK"


def metric_card(
    label: str,
    value: str,
    detail: str,
    status: str,
) -> None:
    """Render a metric card with formal status colouring."""
    normalised = str(status).upper().strip()
    status_label = metric_status_label(normalised)
    st.markdown(
        f"""
        <div class="metric-card {card_class(normalised)}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
            <div class="metric-status">{status_label}</div>
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
    '<div class="dashboard-header">'
    '<div class="dashboard-title">Exeter City Football Club — Solvency Analysis Dashboard</div>'
    '<div class="dashboard-subtitle">'
    'Analysis of short-term liquidity position (13 weeks) and long-term sustainability position (18 months) '
    'with scenario modelling and contribution analysis.'
    '</div>'
    '<div class="disclaimer-text">'
    '⚠️ Proof of Concept — For illustrative purposes only. Subject to validation and formal approval prior to operational use.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Initialize session state for sensitivity factors
if 'revenue_adjustment' not in st.session_state:
    st.session_state.revenue_adjustment = 0
if 'operating_costs_adjustment' not in st.session_state:
    st.session_state.operating_costs_adjustment = 0
if 'cashflow_timing_adjustment' not in st.session_state:
    st.session_state.cashflow_timing_adjustment = 0
if 'probability_adjustment' not in st.session_state:
    st.session_state.probability_adjustment = 0
if 'show_sensitivity' not in st.session_state:
    st.session_state.show_sensitivity = False

# ---------------------------------------------------------------------
# CONTROLS
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="control-label">Model Configuration</div>', unsafe_allow_html=True)
    st.divider()
    
    scenario = st.selectbox("Scenario", ["base"], index=0, help="Select the forecast scenario to analyse")
    entity_view = st.selectbox(
        "Entity View",
        ["Consolidated", "Club", "Trust"],
        help="Consolidated view excludes intercompany transactions; standalone views retain selected entity only",
    )
    
    st.divider()
    st.caption(
        "Consolidated view applies intercompany elimination rules. "
        "Standalone entity views present uneliminated positions for the selected entity."
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
# ASSUMPTIONS - CATEGORY WEIGHTINGS
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

with st.expander("Category Assumption Adjustments", expanded=False):
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write(
        "Modify category weighting assumptions by entering values as whole-number percentages. "
        "For example: **100** for 100%, **75** for 75%, **0** for 0%. "
        "Adjustments are applied without modifying underlying line-item data."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    edited_display_weights = st.data_editor(
        display_weights,
        disabled=["direction", "category", "application_note"],
        column_config={
            "direction": st.column_config.TextColumn("Contributor Type"),
            "category": st.column_config.TextColumn("Category"),
            "weight_3_month": st.column_config.NumberColumn(
                "13-Week Weighting (%)",
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.0f",
                help="Percentage assumption: 100 = 100%",
            ),
            "weight_18_month": st.column_config.NumberColumn(
                "18-Month Weighting (%)",
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.0f",
                help="Percentage assumption: 100 = 100%",
            ),
            "application_note": st.column_config.TextColumn("Application Notes"),
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

# Apply sensitivity adjustments to line items
adjusted_items = filtered_items.copy()
if st.session_state.revenue_adjustment != 0:
    adjusted_items.loc[adjusted_items["direction"].eq("positive"), "gross_amount"] *= (1 + st.session_state.revenue_adjustment / 100)
if st.session_state.operating_costs_adjustment != 0:
    adjusted_items.loc[adjusted_items["direction"].eq("negative"), "gross_amount"] *= (1 + st.session_state.operating_costs_adjustment / 100)
if st.session_state.cashflow_timing_adjustment != 0:
    adjusted_items["relevance_3_month"] = adjusted_items["relevance_3_month"].clip(0, 1) * (1 + st.session_state.cashflow_timing_adjustment / 100)
if st.session_state.probability_adjustment != 0:
    adjusted_items["probability"] = adjusted_items["probability"].clip(0, 1) * (1 + st.session_state.probability_adjustment / 100)

short_detail = calculate_contributions(
    adjusted_items,
    edited_weights,
    "3_month",
    scenario=scenario,
    consolidated=consolidated,
)
long_detail = calculate_contributions(
    adjusted_items,
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
# SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------
st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)

sensitivity_expander = st.expander("Adjust Key Factors to Analyze Sensitivity", expanded=False)

with sensitivity_expander:
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write(
        "Model solvency position under different assumptions. Adjust factors as percentages to see impact on liquidity and sustainability."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sensitivity-label">Revenue / Positive Cash Inflows</div>', unsafe_allow_html=True)
        revenue_adj = st.slider(
            "Revenue adjustment (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.revenue_adjustment,
            step=5,
            label_visibility="collapsed",
            help="Adjust expected revenue by percentage (e.g., -10 = 10% reduction)",
        )
        if revenue_adj != st.session_state.revenue_adjustment:
            st.session_state.revenue_adjustment = revenue_adj
            st.rerun()
        
        st.markdown('<div class="sensitivity-label">Operating Costs / Negative Cash Outflows</div>', unsafe_allow_html=True)
        costs_adj = st.slider(
            "Operating costs adjustment (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.operating_costs_adjustment,
            step=5,
            label_visibility="collapsed",
            help="Adjust expected costs by percentage (e.g., +10 = 10% increase)",
        )
        if costs_adj != st.session_state.operating_costs_adjustment:
            st.session_state.operating_costs_adjustment = costs_adj
            st.rerun()
    
    with col2:
        st.markdown('<div class="sensitivity-label">Cash Flow Timing / Horizon Relevance</div>', unsafe_allow_html=True)
        timing_adj = st.slider(
            "Cashflow timing adjustment (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.cashflow_timing_adjustment,
            step=5,
            label_visibility="collapsed",
            help="Adjust expected cash flow timing by percentage (e.g., -20 = delays reduce near-term availability)",
        )
        if timing_adj != st.session_state.cashflow_timing_adjustment:
            st.session_state.cashflow_timing_adjustment = timing_adj
            st.rerun()
        
        st.markdown('<div class="sensitivity-label">Probability of Realization</div>', unsafe_allow_html=True)
        prob_adj = st.slider(
            "Probability adjustment (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.probability_adjustment,
            step=5,
            label_visibility="collapsed",
            help="Adjust probability weighting by percentage (e.g., -15 = reduce confidence in assumptions)",
        )
        if prob_adj != st.session_state.probability_adjustment:
            st.session_state.probability_adjustment = prob_adj
            st.rerun()
    
    # Display sensitivity summary
    if any([revenue_adj, costs_adj, timing_adj, prob_adj]):
        st.divider()
        st.write("**Sensitivity Summary**")
        sensitivity_summary = []
        if revenue_adj != 0:
            sensitivity_summary.append(f"Revenue: {revenue_adj:+d}%")
        if costs_adj != 0:
            sensitivity_summary.append(f"Operating costs: {costs_adj:+d}%")
        if timing_adj != 0:
            sensitivity_summary.append(f"Cashflow timing: {timing_adj:+d}%")
        if prob_adj != 0:
            sensitivity_summary.append(f"Probability: {prob_adj:+d}%")
        
        st.caption(" | ".join(sensitivity_summary))

# ---------------------------------------------------------------------
# HEADLINE SOLVENCY POSITION
# ---------------------------------------------------------------------
st.markdown('<div class="section-header">Solvency Position Summary</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "13-Week Solvency Ratio",
        f"{short_result.ratio:.2f}x",
        f"Minimum: {short_min:.2f}x | Target: {short_target:.2f}x",
        short_result.status,
    )

with c2:
    metric_card(
        "13-Week Liquidity Headroom",
        f"£{short_result.headroom:,.0f}",
        (
            f"Resources: £{short_result.positives:,.0f} | "
            f"Requirements: £{short_result.negatives:,.0f}"
        ),
        short_result.status,
    )

with c3:
    metric_card(
        "18-Month Solvency Ratio",
        f"{long_result.ratio:.2f}x",
        f"Minimum: {long_min:.2f}x | Target: {long_target:.2f}x",
        long_result.status,
    )

with c4:
    metric_card(
        "18-Month Sustainability Headroom",
        f"£{long_result.headroom:,.0f}",
        (
            f"Resources: £{long_result.positives:,.0f} | "
            f"Requirements: £{long_result.negatives:,.0f}"
        ),
        long_result.status,
    )

st.write("")

# Trend analysis
st.markdown('<div class="section-header">Trend Analysis</div>', unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        trend_chart(
            movement_history,
            "3_month",
            short_min,
            short_target,
            title="13-Week Solvency Ratio — Historical Trend",
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
            title="18-Month Solvency Ratio — Historical Trend",
        ),
        use_container_width=True,
    )

# Category contribution analysis
st.markdown('<div class="section-header">Contribution Analysis by Category</div>', unsafe_allow_html=True)

short_summary = category_summary(short_detail)
long_summary = category_summary(long_detail)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        category_chart(
            short_summary,
            "13-Week Contribution by Category",
        ),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        category_chart(
            long_summary,
            "18-Month Contribution by Category",
        ),
        use_container_width=True,
    )

st.divider()

# Detailed analysis tabs
st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Movement Analysis",
        "Contribution Detail",
        "Line-Item Calculation",
        "Data Validation",
    ]
)

with tab1:
    st.markdown('<div class="tab-description">Movement in liquidity and sustainability headroom between reporting periods</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="tab-description">Eligible contributions aggregated by contributor type and category</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.write("**13-Week Contribution Summary**")
        st.dataframe(
            short_summary,
            hide_index=True,
            use_container_width=True,
        )
    with p2:
        st.write("**18-Month Contribution Summary**")
        st.dataframe(
            long_summary,
            hide_index=True,
            use_container_width=True,
        )

with tab3:
    st.markdown('<div class="tab-description">Line-item calculation detail with applied assumptions and eligible contribution</div>', unsafe_allow_html=True)
    horizon = st.radio(
        "Select calculation horizon:",
        ["3_month", "18_month"],
        horizontal=True,
        format_func=lambda x: "13-Week" if x == "3_month" else "18-Month",
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
        "Download Line-Item Calculation Detail (CSV)",
        detail[display_columns].to_csv(index=False),
        file_name=f"solvency_calculation_{horizon}_detail.csv",
        mime="text/csv",
    )

with tab4:
    st.markdown('<div class="tab-description">Data integrity and consistency validation checks</div>', unsafe_allow_html=True)
    checks = validate_inputs(filtered_items, edited_weights)
    st.dataframe(
        checks,
        hide_index=True,
        use_container_width=True,
    )

    failed = checks["Status"].eq("FAIL").sum()
    if failed:
        st.error(f"⚠️ {failed} validation check(s) failed. Review data quality before using results in decision-making.")
    else:
        st.success("✓ All validation checks passed. Data quality standards met.")
