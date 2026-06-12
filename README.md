# Data-Driven Investment Intelligence Using NIFTY-50 Market Data

This repository implements a starter investment intelligence platform for the NIFTY-50 challenge described in `PS.txt`.

The focus is decision support, not only price prediction:

- Technical indicators: MA, EMA, RSI, MACD, Bollinger Bands, volatility, momentum, and volume spikes.
- Stock predictor engine for next-day and multi-day returns/direction.
- Next trading day buy/sell/hold signal engine.
- Portfolio construction for conservative, balanced, and aggressive profiles.
- Risk assessment using volatility, Sharpe, Sortino, maximum drawdown, VaR, and expected shortfall.
- Explainability, anomaly detection, candlestick history, and investment impact reports.

Because the supplied dataset contains daily historical candles, "24 hrs" is treated as the next available trading day after the latest row in the dataset. The generated signals are educational decision-support indicators, not financial advice.

## Project Structure

```text
.
|-- app/
|   `-- streamlit_app.py
|-- configs/
|   `-- project.yaml
|-- data/
|   `-- processed/
|-- dataset/
|   |-- NIFTY50_all.csv
|   |-- stock_metadata.csv
|   `-- <ticker>.csv
|-- models/
|-- notebooks/
|   |-- 01_preprocess_eda_feature_engineering.ipynb
|   |-- 02_stock_predictor_engine.ipynb
|   |-- 03_portfolio_construction_engine.ipynb
|   |-- 04_risk_assessment_engine.ipynb
|   |-- 05_explainability_anomaly_engine.ipynb
|   |-- 06_next_day_signal_reporting_engine.ipynb
|   `-- 07_neural_network_forecasting_engine.ipynb
|-- reports/
|   `-- figures/
|-- src/
|   `-- invest_intel/
`-- requirements.txt
```

## Dataset Notes

Initial audit of the provided folder:

- Combined file: `dataset/NIFTY50_all.csv`
- Metadata file: `dataset/stock_metadata.csv`
- Combined rows: `235,192`
- Historical symbols in combined file: `65`
- Metadata rows: `50`
- Date range: `2000-01-03` to `2021-04-30`
- High missingness fields: `Trades`, `Deliverable Volume`, `%Deliverble`
- `dataset/INFRATEL.csv` is very small and should be treated as a validation warning.

The 65-vs-50 symbol mismatch is expected for historical index data because symbols change and index membership changes over time.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Notebooks

```bash
jupyter lab
```

Recommended order:

1. `01_preprocess_eda_feature_engineering.ipynb`
2. `02_stock_predictor_engine.ipynb`
3. `03_portfolio_construction_engine.ipynb`
4. `04_risk_assessment_engine.ipynb`
5. `05_explainability_anomaly_engine.ipynb`
6. `06_next_day_signal_reporting_engine.ipynb`
7. `07_neural_network_forecasting_engine.ipynb`

Notebook `06` trains one-day models, creates buy/sell/hold signals, renders candlestick history, and exports `reports/investment_signal_report.csv`.
Notebook `07` contains the lightweight neural-network forecasting baseline.

## Run Prototype App

```bash
streamlit run app/streamlit_app.py
```

The app includes:

- Candlestick and volume history for the selected stock.
- Next trading day signal with confidence and predicted price.
- Portfolio recommendation and allocation.
- Signal leaderboard.
- Downloadable investment impact report.
- Risk leaderboard.

## Rebuild Outputs

```bash
.venv\Scripts\python.exe scripts\run_pipeline.py
```

This regenerates processed features, one-day model artifacts, model metrics, investment signal report, and Markdown summary.

## Smoke Test

```bash
.venv\Scripts\python.exe scripts\smoke_test.py
```

## Reproducibility

All notebooks import reusable code from `src/invest_intel`. Keep major calculations there first, then call them from notebooks and the app.


