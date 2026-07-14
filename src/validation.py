from __future__ import annotations
import pandas as pd
from src.model import REQUIRED_LINE_ITEM_COLUMNS, REQUIRED_WEIGHT_COLUMNS


def validate_inputs(
    line_items: pd.DataFrame,
    weightings: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    missing_items = REQUIRED_LINE_ITEM_COLUMNS - set(line_items.columns)
    add(
        "Required line-item fields",
        not missing_items,
        "Complete" if not missing_items else f"Missing: {sorted(missing_items)}",
    )

    missing_weights = REQUIRED_WEIGHT_COLUMNS - set(weightings.columns)
    add(
        "Required weighting fields",
        not missing_weights,
        "Complete" if not missing_weights else f"Missing: {sorted(missing_weights)}",
    )

    if not missing_items:
        add(
            "Unique record identifiers",
            line_items["record_id"].is_unique,
            "All IDs unique" if line_items["record_id"].is_unique else "Duplicate IDs found",
        )
        add(
            "Contributor directions",
            set(line_items["direction"].str.lower().dropna()).issubset(
                {"positive", "negative"}
            ),
            "Valid" if set(line_items["direction"].str.lower().dropna()).issubset(
                {"positive", "negative"}
            ) else "Only positive and negative are permitted",
        )

        probability = pd.to_numeric(line_items["probability"], errors="coerce")
        add(
            "Probability range",
            probability.between(0, 1).all(),
            "All values between 0 and 1",
        )

        haircut = pd.to_numeric(line_items["liquidity_haircut"], errors="coerce")
        add(
            "Liquidity haircut range",
            haircut.between(0, 1).all(),
            "All values between 0 and 1",
        )

        relevance3 = pd.to_numeric(line_items["relevance_3_month"], errors="coerce")
        relevance18 = pd.to_numeric(line_items["relevance_18_month"], errors="coerce")
        add(
            "Horizon relevance range",
            relevance3.between(0, 1).all() and relevance18.between(0, 1).all(),
            "All relevance values between 0 and 1",
        )

    if not missing_weights:
        w3 = pd.to_numeric(weightings["weight_3_month"], errors="coerce")
        w18 = pd.to_numeric(weightings["weight_18_month"], errors="coerce")
        add(
            "Weighting range",
            w3.between(0, 1).all() and w18.between(0, 1).all(),
            "All weighting values between 0 and 1",
        )
        add(
            "Unique weighting categories",
            not weightings.duplicated(["direction", "category"]).any(),
            "Unique" if not weightings.duplicated(["direction", "category"]).any()
            else "Duplicate category assumptions found",
        )

    return pd.DataFrame(checks)
