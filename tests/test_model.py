import pandas as pd
import pytest

from src.model import (
    calculate_contributions,
    calculate_solvency,
    classify_status,
)


def sample_items():
    return pd.DataFrame(
        [
            {
                "record_id": "1",
                "entity": "Club",
                "source_statement": "Balance Sheet",
                "source_account": "1000",
                "line_item": "Cash",
                "reporting_date": "2026-06-30",
                "forecast_period": "2026-06-30",
                "due_date": "2026-06-30",
                "scenario": "base",
                "gross_amount": 100,
                "direction": "positive",
                "category": "Cash",
                "probability": 1,
                "liquidity_haircut": 0,
                "discount_factor": 1,
                "relevance_3_month": 1,
                "relevance_18_month": 1,
                "intercompany_flag": False,
                "data_quality_status": "Approved",
            },
            {
                "record_id": "2",
                "entity": "Club",
                "source_statement": "Cash Flow",
                "source_account": "2000",
                "line_item": "Payroll",
                "reporting_date": "2026-06-30",
                "forecast_period": "2026-09-30",
                "due_date": "2026-09-30",
                "scenario": "base",
                "gross_amount": 80,
                "direction": "negative",
                "category": "Payroll",
                "probability": 1,
                "liquidity_haircut": 0,
                "discount_factor": 1,
                "relevance_3_month": 1,
                "relevance_18_month": 1,
                "intercompany_flag": False,
                "data_quality_status": "Approved",
            },
        ]
    )


def sample_weights():
    return pd.DataFrame(
        [
            {
                "direction": "positive",
                "category": "Cash",
                "weight_3_month": 1,
                "weight_18_month": 1,
            },
            {
                "direction": "negative",
                "category": "Payroll",
                "weight_3_month": 1,
                "weight_18_month": 1,
            },
        ]
    )


def test_basic_solvency_calculation():
    detail = calculate_contributions(
        sample_items(), sample_weights(), "3_month", "base", True
    )
    result = calculate_solvency(detail, "3_month", 1.0, 1.25)
    assert result.positives == pytest.approx(100)
    assert result.negatives == pytest.approx(80)
    assert result.ratio == pytest.approx(1.25)
    assert result.headroom == pytest.approx(20)
    assert result.status == "GREEN"


def test_weighting_changes_contribution():
    weights = sample_weights()
    weights.loc[weights["category"].eq("Cash"), "weight_3_month"] = 0.5
    detail = calculate_contributions(
        sample_items(), weights, "3_month", "base", True
    )
    result = calculate_solvency(detail, "3_month", 1.0, 1.25)
    assert result.positives == pytest.approx(50)
    assert result.ratio == pytest.approx(0.625)
    assert result.status == "RED"


def test_status_classification():
    assert classify_status(0.8, 1.0, 1.25) == "RED"
    assert classify_status(1.1, 1.0, 1.25) == "AMBER"
    assert classify_status(1.3, 1.0, 1.25) == "GREEN"
