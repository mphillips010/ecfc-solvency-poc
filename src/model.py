from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd

Horizon = Literal["3_month", "18_month"]


@dataclass(frozen=True)
class SolvencyResult:
    horizon: Horizon
    positives: float
    negatives: float
    ratio: float
    headroom: float
    status: str
    minimum_threshold: float
    target_threshold: float


REQUIRED_LINE_ITEM_COLUMNS = {
    "record_id",
    "entity",
    "source_statement",
    "source_account",
    "line_item",
    "reporting_date",
    "forecast_period",
    "due_date",
    "scenario",
    "gross_amount",
    "direction",
    "category",
    "probability",
    "liquidity_haircut",
    "discount_factor",
    "relevance_3_month",
    "relevance_18_month",
    "intercompany_flag",
    "data_quality_status",
}

REQUIRED_WEIGHT_COLUMNS = {
    "direction",
    "category",
    "weight_3_month",
    "weight_18_month",
}


def prepare_line_items(line_items: pd.DataFrame) -> pd.DataFrame:
    df = line_items.copy()
    missing = REQUIRED_LINE_ITEM_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing line-item columns: {sorted(missing)}")

    for column in [
        "gross_amount",
        "probability",
        "liquidity_haircut",
        "discount_factor",
        "relevance_3_month",
        "relevance_18_month",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for date_column in ["reporting_date", "forecast_period", "due_date"]:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    df["direction"] = df["direction"].str.lower().str.strip()
    df["scenario"] = df["scenario"].str.lower().str.strip()
    df["intercompany_flag"] = (
        df["intercompany_flag"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    )

    return df


def prepare_weightings(weightings: pd.DataFrame) -> pd.DataFrame:
    df = weightings.copy()
    missing = REQUIRED_WEIGHT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing weighting columns: {sorted(missing)}")

    for column in ["weight_3_month", "weight_18_month"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["direction"] = df["direction"].str.lower().str.strip()
    return df


def calculate_contributions(
    line_items: pd.DataFrame,
    weightings: pd.DataFrame,
    horizon: Horizon,
    scenario: str = "base",
    consolidated: bool = True,
) -> pd.DataFrame:
    items = prepare_line_items(line_items)
    weights = prepare_weightings(weightings)

    items = items[items["scenario"].eq(scenario.lower())].copy()

    if consolidated:
        items = items[~items["intercompany_flag"]].copy()

    weight_column = "weight_3_month" if horizon == "3_month" else "weight_18_month"
    relevance_column = (
        "relevance_3_month" if horizon == "3_month" else "relevance_18_month"
    )

    merged = items.merge(
        weights[["direction", "category", weight_column]],
        on=["direction", "category"],
        how="left",
        validate="many_to_one",
    )

    if merged[weight_column].isna().any():
        categories = merged.loc[
            merged[weight_column].isna(), ["direction", "category"]
        ].drop_duplicates()
        raise ValueError(
            "Missing category weightings for: "
            + ", ".join(
                f"{row.direction}/{row.category}"
                for row in categories.itertuples(index=False)
            )
        )

    merged["eligible_contribution"] = (
        merged["gross_amount"].abs()
        * merged[relevance_column].clip(0, 1)
        * merged[weight_column].clip(0, 1)
        * merged["probability"].clip(0, 1)
        * (1 - merged["liquidity_haircut"].clip(0, 1))
        * merged["discount_factor"].clip(lower=0)
    )

    merged["signed_contribution"] = np.where(
        merged["direction"].eq("positive"),
        merged["eligible_contribution"],
        -merged["eligible_contribution"],
    )

    merged["applied_weight"] = merged[weight_column]
    merged["applied_relevance"] = merged[relevance_column]
    merged["horizon"] = horizon
    return merged


def classify_status(
    ratio: float,
    minimum_threshold: float,
    target_threshold: float,
) -> str:
    if not np.isfinite(ratio):
        return "GREEN" if ratio > 0 else "RED"
    if ratio < minimum_threshold:
        return "RED"
    if ratio < target_threshold:
        return "AMBER"
    return "GREEN"


def calculate_solvency(
    contributions: pd.DataFrame,
    horizon: Horizon,
    minimum_threshold: float,
    target_threshold: float,
) -> SolvencyResult:
    positives = contributions.loc[
        contributions["direction"].eq("positive"), "eligible_contribution"
    ].sum()
    negatives = contributions.loc[
        contributions["direction"].eq("negative"), "eligible_contribution"
    ].sum()

    ratio = np.inf if negatives == 0 and positives > 0 else (
        0.0 if negatives == 0 else positives / negatives
    )
    headroom = positives - negatives
    status = classify_status(ratio, minimum_threshold, target_threshold)

    return SolvencyResult(
        horizon=horizon,
        positives=float(positives),
        negatives=float(negatives),
        ratio=float(ratio),
        headroom=float(headroom),
        status=status,
        minimum_threshold=float(minimum_threshold),
        target_threshold=float(target_threshold),
    )


def category_summary(contributions: pd.DataFrame) -> pd.DataFrame:
    return (
        contributions.groupby(["direction", "category"], as_index=False)
        .agg(
            gross_amount=("gross_amount", lambda s: s.abs().sum()),
            eligible_contribution=("eligible_contribution", "sum"),
        )
        .sort_values(["direction", "eligible_contribution"], ascending=[True, False])
    )


def movement_bridge(history: pd.DataFrame, horizon: Horizon) -> pd.DataFrame:
    df = history.copy()
    df["reporting_date"] = pd.to_datetime(df["reporting_date"], errors="coerce")
    df = df[df["horizon"].eq(horizon)].sort_values("reporting_date")

    if df.empty:
        return df

    latest_date = df["reporting_date"].max()
    previous_dates = sorted(df.loc[df["reporting_date"] < latest_date, "reporting_date"].unique())
    if not previous_dates:
        return df[df["reporting_date"].eq(latest_date)]

    previous_date = previous_dates[-1]
    return df[df["reporting_date"].isin([previous_date, latest_date])].copy()
