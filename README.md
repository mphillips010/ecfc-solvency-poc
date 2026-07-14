# ECFC Solvency Proof of Concept

A GitHub-ready proof of concept for calculating and reporting:

- **13-week short-term solvency**
- **18-month long-term sustainability**
- trend analysis
- movement analysis
- scenario analysis
- contribution by category and line item
- data-quality and reconciliation checks

The application is implemented in Python using Streamlit.

## Core calculation

For each line item:

```text
eligible contribution
= gross amount
× horizon relevance
× category weighting
× probability
× (1 - liquidity haircut)
× discount factor
```

Positive and negative contributors are calculated separately.

```text
solvency ratio
= total eligible positive contributions
÷ total eligible negative contributions
```

```text
headroom
= total eligible positive contributions
- total eligible negative contributions
```

## Project structure

```text
ecfc-solvency-poc/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── line_items.csv
│   ├── category_weightings.csv
│   ├── thresholds.csv
│   └── movement_history.csv
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── validation.py
│   └── charts.py
└── tests/
    └── test_model.py
```

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Data model

`data/line_items.csv` is the underlying item-level input. Each record includes:

- entity
- source statement
- source account
- reporting date
- forecast period
- due date
- scenario
- gross amount
- contributor direction
- category
- probability
- liquidity haircut
- horizon relevance
- intercompany flag
- data-quality status

`data/category_weightings.csv` contains the editable category assumptions. The dashboard allows these assumptions to be changed without changing the line-item data.

## Important proof-of-concept limitations

This repository is illustrative and does not contain Exeter City’s actual financial data.

Before operational use:

1. connect the model to approved accounting and forecasting data;
2. approve definitions, thresholds and category weightings;
3. validate intercompany eliminations;
4. test contractual timing and forecast assumptions;
5. independently review methodology and controls;
6. add authentication, access controls and formal change governance.

## Suggested next development steps

- database or accounting-system integration;
- import templates for trial balance and cash-flow forecasts;
- persistent scenario and assumption versions;
- user authentication and approval workflow;
- Power BI or embedded Board-pack output;
- formal audit logging;
- reverse stress testing;
- automated monthly snapshotting.
