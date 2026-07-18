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
    page_title="ECFC Solvency Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# COMPLETE VISUAL REDESIGN - FIRST PRINCIPLES
# Clean, accessible, professional dashboard with ECFC brand elements
# Dark theme for primary content, light backgrounds for data readability
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap');

        * {
            margin: 0;
            padding: 0;
        }

        html, body {
            background-color: #ffffff;
            color: #1f2937;
            font-family: 'Barlow', 'Segoe UI', Arial, sans-serif;
        }

        .block-container {
            max-width: 1400px;
            padding: 2rem 1.5rem;
        }

        /* Main header - dark bar with gold accent */
        .page-header {
            background-color: #111827;
            color: #ffffff;
            padding: 2.5rem 2rem;
            margin: -2rem -1.5rem 2.5rem -1.5rem;
            border-bottom: 4px solid #D7B477;
        }

        .header-title {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }

        .header-title-accent {
            color: #D7B477;
        }

        .header-subtitle {
            font-size: 0.95rem;
            color: #d1d5db;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .header-campaign {
            font-size: 0.85rem;
            color: #D7B477;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.75rem;
        }

        .header-warning {
            background-color: rgba(215, 180, 119, 0.1);
            border-left: 3px solid #D7B477;
            padding: 0.75rem 1rem;
            margin-top: 1rem;
            font-size: 0.85rem;
            color: #d1d5db;
            border-radius: 3px;
        }

        /* Section headers */
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #111827;
            margin: 2.5rem 0 1.5rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid #D7B477;
        }

        /* Metric cards - clean white with border accents */
        .metric-card {
            background-color: #ffffff;
            border: 2px solid #e5e7eb;
            border-radius: 6px;
            padding: 1.5rem;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            transition: border-color 0.2s ease;
        }

        .metric-card:hover {
            border-color: #D7B477;
        }

        .metric-card.status-compliant {
            border-left: 4px solid #059669;
        }

        .metric-card.status-caution {
            border-left: 4px solid #d97706;
        }

        .metric-card.status-critical {
            border-left: 4px solid #dc2626;
        }

        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #D7B477;
            margin-bottom: 0.5rem;
            font-family: 'Barlow', monospace;
        }

        .metric-detail {
            font-size: 0.8rem;
            line-height: 1.5;
            color: #6b7280;
            margin-bottom: auto;
        }

        .metric-status {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.4rem 0.75rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            width: fit-content;
            background-color: #f3f4f6;
            color: #1f2937;
            border: 1px solid #d1d5db;
        }

        .metric-card.status-compliant .metric-status {
            background-color: #d1fae5;
            color: #065f46;
            border-color: #6ee7b7;
        }

        .metric-card.status-caution .metric-status {
            background-color: #fef3c7;
            color: #78350f;
            border-color: #fcd34d;
        }

        .metric-card.status-critical .metric-status {
            background-color: #fee2e2;
            color: #7f1d1d;
            border-color: #fca5a5;
        }

        /* Control panel */
        .control-panel {
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }

        .control-label {
            font-size: 0.9rem;
            font-weight: 700;
            color: #111827;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.75rem;
            display: block;
        }

        .control-sublabel {
            font-size: 0.8rem;
            font-weight: 600;
            color: #374151;
            margin-top: 1.25rem;
            margin-bottom: 0.5rem;
        }

        /* Data visualization containers */
        .chart-container {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .data-table {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
        }

        /* Info boxes */
        .info-box {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 3px;
            font-size: 0.9rem;
            color: #1e40af;
        }

        .info-box.warning {
            background-color: #fef3c7;
            border-left-color: #d97706;
            color: #78350f;
        }

        .info-box.error {
            background-color: #fee2e2;
            border-left-color: #dc2626;
            color: #7f1d1d;
        }

        .info-box.success {
            background-color: #dcfce7;
            border-left-color: #059669;
            color: #065f46;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] button {
            font-weight: 600;
            color: #6b7280;
        }

        .stTabs [aria-selected="true"] {
            color: #111827 !important;
            border-bottom-color: #D7B477 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f9fafb;
            border-right: 1px solid #e5e7eb;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }

        /* Dividers */
        hr {
            background-color: #e5e7eb;
            border: none;
            height: 1px;
            margin: 1.5rem 0;
        }

        /* Text utilities */
        .text-muted {
            color: #6b7280;
        }

        .text-accent {
            color: #D7B477;
            font-weight: 600;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .page-header {
                padding: 1.5rem;
            }

            .header-title {
                font-size: 1.5rem;
            }

            .metric-value {
                font-size: 1.5rem;
            }

            .block-container {
                padding: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_card_class(status: str) -> str:
    """Return card class based on status."""
    normalised = str(status).upper().strip()
    if normalised == "GREEN":
        return "status-compliant"
    if normalised == "RED":
        return "status-critical"
    return "status-caution"


def get_status_label(status: str) -> str:
    """Return formal status label."""
    normalised = str(status).upper().strip()
    if normalised == "GREEN":
        return "COMPLIANT"
    if normalised == "RED":
        return "NON-COMPLIANT"
    return "AT RISK"


def render_metric_card(label: str, value: str, detail: str, status: str) -> None:
    """Render a metric card."""
    card_class = get_card_class(status)
    status_label = get_status_label(status)
    
    st.markdown(
        f"""
        <div class="metric-card {card_class}">
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
    """Load all required data files."""
    return (
        pd.read_csv(DATA_DIR / "line_items.csv"),
        pd.read_csv(DATA_DIR / "category_weightings.csv"),
        pd.read_csv(DATA_DIR / "thresholds.csv"),
        pd.read_csv(DATA_DIR / "movement_history.csv"),
    )


# Load data
line_items, default_weights, thresholds, movement_history = load_data()

# Render page header
st.markdown(
    """
    <div class="page-header">
        <div class="header-title">Exeter City Football Club</div>
        <div class="header-title" style="font-size: 1.5rem; color: #D7B477;">Solvency Analysis Dashboard</div>
        <div class="header-subtitle">
            Financial analysis tool for monitoring short-term liquidity (13 weeks) and long-term sustainability (18 months).
        </div>
        <div class="header-campaign">Community-driven • Academy-powered • Supporter first</div>
        <div class="header-warning">
            ⚠️ Proof of Concept. For illustrative purposes only. Subject to validation and formal approval prior to operational use.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'revenue_adjustment' not in st.session_state:
    st.session_state.revenue_adjustment = 0
if 'operating_costs_adjustment' not in st.session_state:
    st.session_state.operating_costs_adjustment = 0
if 'cashflow_timing_adjustment' not in st.session_state:
    st.session_state.cashflow_timing_adjustment = 0
if 'probability_adjustment' not in st.session_state:
    st.session_state.probability_adjustment = 0

# Sidebar configuration
with st.sidebar:
    st.markdown("### Configuration")
    scenario = st.selectbox("Scenario", ["base"], index=0)
    entity_view = st.selectbox(
        "Entity View",
        ["Consolidated", "Club", "Trust"],
    )
    st.divider()
    st.caption(
        "**Consolidated**: Excludes intercompany transactions. "
        "**Standalone**: Shows uneliminated entity position."
    )

# Determine filtered items and consolidation flag
if entity_view == "Club":
    filtered_items = line_items[line_items["entity"].eq("Club")].copy()
    consolidated = False
elif entity_view == "Trust":
    filtered_items = line_items[line_items["entity"].eq("Trust")].copy()
    consolidated = False
else:
    filtered_items = line_items.copy()
    consolidated = True

# Category weightings
display_weights = default_weights.copy()
display_weights["weight_3_month"] = (
    pd.to_numeric(display_weights["weight_3_month"], errors="coerce") * 100
)
display_weights["weight_18_month"] = (
    pd.to_numeric(display_weights["weight_18_month"], errors="coerce") * 100
)

with st.expander("📋 Adjust Category Assumptions", expanded=False):
    st.info(
        "Modify weighting assumptions as percentages (0–100). Adjustments apply without changing underlying data."
    )
    
    edited_display_weights = st.data_editor(
        display_weights,
        disabled=["direction", "category", "application_note"],
        column_config={
            "direction": st.column_config.TextColumn("Contributor Type", width=120),
            "category": st.column_config.TextColumn("Category", width=120),
            "weight_3_month": st.column_config.NumberColumn(
                "13-Week %",
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.0f",
            ),
            "weight_18_month": st.column_config.NumberColumn(
                "18-Month %",
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.0f",
            ),
            "application_note": st.column_config.TextColumn("Notes", width=150),
        },
        hide_index=True,
        use_container_width=True,
        key="category_weighting_editor",
    )
    
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

# Apply sensitivity adjustments
adjusted_items = filtered_items.copy()
if st.session_state.revenue_adjustment != 0:
    adjusted_items.loc[adjusted_items["direction"].eq("positive"), "gross_amount"] *= (
        1 + st.session_state.revenue_adjustment / 100
    )
if st.session_state.operating_costs_adjustment != 0:
    adjusted_items.loc[adjusted_items["direction"].eq("negative"), "gross_amount"] *= (
        1 + st.session_state.operating_costs_adjustment / 100
    )
if st.session_state.cashflow_timing_adjustment != 0:
    adjusted_items["relevance_3_month"] = (
        adjusted_items["relevance_3_month"].clip(0, 1) * (1 + st.session_state.cashflow_timing_adjustment / 100)
    )
if st.session_state.probability_adjustment != 0:
    adjusted_items["probability"] = (
        adjusted_items["probability"].clip(0, 1) * (1 + st.session_state.probability_adjustment / 100)
    )

# Calculate contributions and solvency
short_detail = calculate_contributions(
    adjusted_items, edited_weights, "3_month", scenario=scenario, consolidated=consolidated
)
long_detail = calculate_contributions(
    adjusted_items, edited_weights, "18_month", scenario=scenario, consolidated=consolidated
)

short_result = calculate_solvency(short_detail, "3_month", short_min, short_target)
long_result = calculate_solvency(long_detail, "18_month", long_min, long_target)

# Sensitivity Analysis Section
st.markdown('<div class="section-title">Sensitivity Analysis</div>', unsafe_allow_html=True)

with st.expander("🎯 Adjust Key Factors", expanded=False):
    st.info("Model solvency under different assumptions. Adjust factors to see real-time impact.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="control-sublabel">Revenue Impact</div>', unsafe_allow_html=True)
        revenue_adj = st.slider(
            "Revenue / Positive Inflows (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.revenue_adjustment,
            step=5,
            label_visibility="collapsed",
        )
        if revenue_adj != st.session_state.revenue_adjustment:
            st.session_state.revenue_adjustment = revenue_adj
            st.rerun()
        
        st.markdown('<div class="control-sublabel">Cost Impact</div>', unsafe_allow_html=True)
        costs_adj = st.slider(
            "Operating Costs / Outflows (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.operating_costs_adjustment,
            step=5,
            label_visibility="collapsed",
        )
        if costs_adj != st.session_state.operating_costs_adjustment:
            st.session_state.operating_costs_adjustment = costs_adj
            st.rerun()
    
    with col2:
        st.markdown('<div class="control-sublabel">Timing Impact</div>', unsafe_allow_html=True)
        timing_adj = st.slider(
            "Cashflow Timing (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.cashflow_timing_adjustment,
            step=5,
            label_visibility="collapsed",
        )
        if timing_adj != st.session_state.cashflow_timing_adjustment:
            st.session_state.cashflow_timing_adjustment = timing_adj
            st.rerun()
        
        st.markdown('<div class="control-sublabel">Probability Impact</div>', unsafe_allow_html=True)
        prob_adj = st.slider(
            "Probability of Realization (%)",
            min_value=-50,
            max_value=50,
            value=st.session_state.probability_adjustment,
            step=5,
            label_visibility="collapsed",
        )
        if prob_adj != st.session_state.probability_adjustment:
            st.session_state.probability_adjustment = prob_adj
            st.rerun()
    
    if any([revenue_adj, costs_adj, timing_adj, prob_adj]):
        st.divider()
        summary_items = []
        if revenue_adj != 0:
            summary_items.append(f"Revenue {revenue_adj:+d}%")
        if costs_adj != 0:
            summary_items.append(f"Costs {costs_adj:+d}%")
        if timing_adj != 0:
            summary_items.append(f"Timing {timing_adj:+d}%")
        if prob_adj != 0:
            summary_items.append(f"Probability {prob_adj:+d}%")
        st.caption(f"**Active adjustments**: {' • '.join(summary_items)}")

# Solvency Position Section
st.markdown('<div class="section-title">Solvency Position</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        "13-Week Ratio",
        f"{short_result.ratio:.2f}x",
        f"Min: {short_min:.2f}x • Target: {short_target:.2f}x",
        short_result.status,
    )

with col2:
    render_metric_card(
        "13-Week Headroom",
        f"£{short_result.headroom:,.0f}",
        f"Resources: £{short_result.positives:,.0f}\nRequirements: £{short_result.negatives:,.0f}",
        short_result.status,
    )

with col3:
    render_metric_card(
        "18-Month Ratio",
        f"{long_result.ratio:.2f}x",
        f"Min: {long_min:.2f}x • Target: {long_target:.2f}x",
        long_result.status,
    )

with col4:
    render_metric_card(
        "18-Month Headroom",
        f"£{long_result.headroom:,.0f}",
        f"Resources: £{long_result.positives:,.0f}\nRequirements: £{long_result.negatives:,.0f}",
        long_result.status,
    )

# Trends Section
st.markdown('<div class="section-title">Historical Trends</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        trend_chart(
            movement_history,
            "3_month",
            short_min,
            short_target,
            title="13-Week Solvency Ratio",
        ),
        use_container_width=True,
    )

with col2:
    st.plotly_chart(
        trend_chart(
            movement_history,
            "18_month",
            long_min,
            long_target,
            title="18-Month Solvency Ratio",
        ),
        use_container_width=True,
    )

# Contributions Section
st.markdown('<div class="section-title">Contribution Analysis</div>', unsafe_allow_html=True)

short_summary = category_summary(short_detail)
long_summary = category_summary(long_detail)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        category_chart(short_summary, "13-Week Contributions"),
        use_container_width=True,
    )

with col2:
    st.plotly_chart(
        category_chart(long_summary, "18-Month Contributions"),
        use_container_width=True,
    )

st.divider()

# Detailed Analysis Tabs
st.markdown('<div class="section-title">Detailed Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Movement", "Contributions", "Calculations", "Validation"]
)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            movement_waterfall(movement_history, "3_month"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            movement_waterfall(movement_history, "18_month"),
            use_container_width=True,
        )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("13-Week")
        st.dataframe(short_summary, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("18-Month")
        st.dataframe(long_summary, use_container_width=True, hide_index=True)

with tab3:
    horizon = st.radio("Horizon:", ["3_month", "18_month"], horizontal=True)
    detail = short_detail if horizon == "3_month" else long_detail
    
    display_columns = [
        "record_id", "entity", "source_statement", "category", "gross_amount",
        "applied_weight", "probability", "eligible_contribution",
        "data_quality_status",
    ]
    
    st.dataframe(
        detail[display_columns].sort_values(
            ["category", "eligible_contribution"], ascending=[True, False]
        ),
        use_container_width=True,
        hide_index=True,
    )
    
    st.download_button(
        "Download Detail (CSV)",
        detail[display_columns].to_csv(index=False),
        file_name=f"solvency_{horizon}.csv",
        mime="text/csv",
    )

with tab4:
    checks = validate_inputs(filtered_items, edited_weights)
    st.dataframe(checks, use_container_width=True, hide_index=True)
    
    failed = checks["Status"].eq("FAIL").sum()
    if failed:
        st.markdown(
            f'<div class="info-box error">⚠️ {failed} check(s) failed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box success">✓ All checks passed</div>',
            unsafe_allow_html=True,
        )
