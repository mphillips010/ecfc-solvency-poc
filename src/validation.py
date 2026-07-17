from __future__ import annotations
import pandas as pd
from src.model import REQUIRED_LINE_ITEM_COLUMNS, REQUIRED_WEIGHT_COLUMNS


def validate_inputs(
    line_items: pd.DataFrame,
    weightings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform comprehensive validation of input data and assumptions.
    
    Returns a DataFrame with validation results including:
    - Check description
    - Pass/Fail status
    - Supporting detail
    """
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        """Record validation check result."""
        checks.append(
            {
                "Check": name,
                "Status": "PASS" if passed else "FAIL",
                "Detail": detail,
            }
        )

    # Line-item schema validation
    missing_items = REQUIRED_LINE_ITEM_COLUMNS - set(line_items.columns)
    add(
        "Line-item data structure",
        not missing_items,
        "All required fields present" if not missing_items else f"Missing fields: {', '.join(sorted(missing_items))}",
    )

    # Weighting schema validation
    missing_weights = REQUIRED_WEIGHT_COLUMNS - set(weightings.columns)
    add(
        "Category weighting structure",
        not missing_weights,
        "All required fields present" if not missing_weights else f"Missing fields: {', '.join(sorted(missing_weights))}",
    )

    # Line-item integrity checks
    if not missing_items:
        add(
            "Record identifier uniqueness",
            line_items["record_id"].is_unique,
            "All record IDs are unique" if line_items["record_id"].is_unique else "Duplicate record IDs detected",
        )
        
        add(
            "Contributor direction values",
            set(line_items["direction"].str.lower().dropna()).issubset(
                {"positive", "negative"}
            ),
            "Valid (positive/negative only)" if set(line_items["direction"].str.lower().dropna()).issubset(
                {"positive", "negative"}
            ) else "Invalid values detected (only 'positive' and 'negative' permitted)",
        )

        probability = pd.to_numeric(line_items["probability"], errors="coerce")
        add(
            "Probability value range",
            probability.between(0, 1).all(),
            "All values within valid range [0, 1]" if probability.between(0, 1).all() else "Values outside range [0, 1] detected",
        )

        haircut = pd.to_numeric(line_items["liquidity_haircut"], errors="coerce")
        add(
            "Liquidity haircut range",
            haircut.between(0, 1).all(),
            "All values within valid range [0, 1]" if haircut.between(0, 1).all() else "Values outside range [0, 1] detected",
        )

        relevance3 = pd.to_numeric(line_items["relevance_3_month"], errors="coerce")
        relevance18 = pd.to_numeric(line_items["relevance_18_month"], errors="coerce")
        add(
            "Horizon relevance range",
            relevance3.between(0, 1).all() and relevance18.between(0, 1).all(),
            "All relevance values within valid range [0, 1]" if (relevance3.between(0, 1).all() and relevance18.between(0, 1).all()) else "Relevance values outside range [0, 1] detected",
        )

    # Category weighting integrity checks
    if not missing_weights:
        w3 = pd.to_numeric(weightings["weight_3_month"], errors="coerce")
        w18 = pd.to_numeric(weightings["weight_18_month"], errors="coerce")
        add(
            "Category weighting range",
            w3.between(0, 1).all() and w18.between(0, 1).all(),
            "All weighting values within valid range [0, 1]" if (w3.between(0, 1).all() and w18.between(0, 1).all()) else "Weighting values outside range [0, 1] detected",
        )
        
        add(
            "Category weighting uniqueness",
            not weightings.duplicated(["direction", "category"]).any(),
            "All direction/category combinations unique" if not weightings.duplicated(["direction", "category"]).any()
            else "Duplicate direction/category combinations detected",
        )

    return pd.DataFrame(checks)
