# Technical Report Draft

## 1. Objective

Build an AI-powered NIFTY-50 investment intelligence platform using only the provided historical market dataset. The solution supports stock behavior forecasting, buy/sell/hold indication, portfolio construction, risk assessment, explainability, anomaly detection, and reproducible reporting.

## 2. Dataset

The project uses `dataset/NIFTY50_all.csv` and `dataset/stock_metadata.csv`.

- Historical rows: 235,192
- Historical symbols: 65
- Metadata symbols: 50
- Date range: 2000-01-03 to 2021-04-30
- Known missing fields: `Trades`, `Deliverable Volume`, `%Deliverble`

Historical symbol count differs from metadata count because index membership and ticker names changed over time.

## 3. Feature Engineering

Features are computed per symbol to avoid cross-stock leakage:

- Daily return and log return
- Moving averages: 5, 20, 50, 100 days
- Close-to-moving-average ratios
- Exponential moving averages: 12 and 26 days
- MACD and MACD signal
- RSI with 14-day lookback
- Bollinger upper/lower bands and band width
- Rolling volatility: 10, 20, 60 days
- Momentum: 10, 20, 60 days
- Volume z-score
- Forward return targets for 1, 5, and 21 trading days
- Direction targets for 1, 5, and 21 trading days

## 4. Stock Predictor Engine

The predictor engine uses time-based validation from 2019 onward.

Implemented models:

- Ridge regression baseline
- HistGradientBoostingRegressor
- RandomForestClassifier for direction
- MLPRegressor neural-network baseline

Metrics:

- MAE
- RMSE
- R2
- Directional accuracy

Daily forecasting is noisy and close to efficient-market behavior, so predictions are combined with technical confirmation before producing buy/sell/hold signals.

## 5. Next Trading Day Signal Engine

Because the dataset has daily candles, the "24 hrs" signal is interpreted as the next available trading day.

Signal inputs:

- Model-predicted next-day return
- MA trend confirmation
- RSI overbought/oversold state
- MACD direction
- Bollinger position
- Momentum
- Short-term volatility
- Volume spike flag

Output labels:

- STRONG BUY
- BUY
- HOLD
- SELL
- STRONG SELL

Each signal includes confidence, predicted return, predicted price, and explanation text.

## 6. Portfolio Construction

Profiles:

- Conservative: lower concentration, inverse-volatility tilt
- Balanced: risk-adjusted score weighting
- Aggressive: stronger tilt toward high Sharpe candidates

The allocation logic caps single-stock exposure and normalizes weights to 100%.

## 7. Risk Assessment

Risk metrics:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- 95% Value at Risk
- 95% Expected Shortfall

Portfolio impact score combines allocation weight, predicted next-day return, and confidence.

## 8. Explainability and Anomaly Detection

Explainability:

- Feature importance for models that expose coefficients or feature importances
- Signal reasons from technical indicators
- Portfolio justification using expected return, volatility, and Sharpe score

Anomaly detection:

- Return shocks using rolling return z-score
- Unusual volume using rolling volume z-score

## 9. Prototype

The Streamlit app provides:

- Next-day signal dashboard
- Candlestick and volume history
- Portfolio allocation view
- Risk-return map
- Anomaly explorer
- Downloadable CSV and Markdown reports

## 10. Limitations

- No live data, news, sentiment, or external APIs are used.
- The dataset ends on 2021-04-30.
- Signals are based on historical daily data and are not intraday trading calls.
- Daily return prediction is inherently noisy; outputs should be treated as decision-support evidence, not guaranteed recommendations.
