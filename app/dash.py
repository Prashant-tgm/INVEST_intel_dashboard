from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from invest_intel.data import attach_metadata, build_returns_matrix, load_prices
from invest_intel.explainability import detect_market_anomalies
from invest_intel.features import add_technical_features
from invest_intel.forecasting import exponential_next_day_forecast, statistical_next_day_forecast
from invest_intel.portfolio import construct_profile_portfolio
from invest_intel.reporting import build_investment_report, build_markdown_summary
from invest_intel.risk import risk_summary
from invest_intel.signals import signals_from_saved_model
from invest_intel.visualization import candlestick_with_volume


st.set_page_config(page_title="NIFTY-50 Investment Intelligence", layout="wide")
st.title("NIFTY-50 Investment Intelligence")
st.caption("Next-trading-day signals from historical daily data. Educational decision support, not financial advice.")


@st.cache_data(show_spinner=False)
def load_app_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = attach_metadata(load_prices())
    features = add_technical_features(prices)
    return prices, features


@st.cache_data(show_spinner=False)
def build_app_outputs(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    start_date: pd.Timestamp,
    profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recent_prices = prices[prices["date"] >= start_date]
    returns = build_returns_matrix(recent_prices, min_obs=126)
    risk = risk_summary(returns)
    weights, metrics, selected = construct_profile_portfolio(returns.dropna(how="all"), profile=profile)
    signals = signals_from_saved_model(features)
    report = build_investment_report(signals, risk, selected)
    stat_forecast = statistical_next_day_forecast(recent_prices)
    exp_forecast = exponential_next_day_forecast(recent_prices)
    return signals, risk, selected, metrics, report, stat_forecast, exp_forecast


prices, features = load_app_data()
symbols = sorted(prices["symbol"].unique())

with st.sidebar:
    default_symbol = symbols.index("RELIANCE") if "RELIANCE" in symbols else 0
    symbol = st.selectbox("Symbol", symbols, index=default_symbol)
    profile = st.selectbox("Investor profile", ["conservative", "balanced", "aggressive"], index=1)
    start_date = st.date_input("History start", value=pd.Timestamp("2015-01-01"))
    top_n = st.slider("Rows", min_value=10, max_value=50, value=25, step=5)

start_timestamp = pd.Timestamp(start_date)
signals, risk, selected, metrics, report, stat_forecast, exp_forecast = build_app_outputs(
    prices,
    features,
    start_timestamp,
    profile,
)
selected_signal = signals[signals["symbol"] == symbol]
latest_date = prices["date"].max().date()
using_indicator_fallback = set(signals["signal_source"].dropna().unique()) == {"technical_indicators"}

overview_tab, history_tab, portfolio_tab, risk_tab, anomaly_tab, report_tab = st.tabs(
    ["Signals", "History", "Portfolio", "Risk", "Anomalies", "Report"]
)

with overview_tab:
    if using_indicator_fallback:
        st.warning(
            "Model artifact loading is unavailable, so signals are being generated from technical indicators only."
        )

    metric_cols = st.columns(5)
    if selected_signal.empty:
        metric_cols[0].metric("Action", "N/A")
        metric_cols[1].metric("Confidence", "N/A")
        metric_cols[2].metric("Predicted return", "N/A")
        metric_cols[3].metric("Close", "N/A")
        metric_cols[4].metric("Next price", "N/A")
    else:
        row = selected_signal.iloc[0]
        metric_cols[0].metric("Action", row["signal"])
        metric_cols[1].metric("Confidence", f"{row['confidence']:.0%}")
        metric_cols[2].metric("Predicted return", f"{row['predicted_return_1d']:.2%}")
        metric_cols[3].metric("Close", f"{row['close']:.2f}")
        metric_cols[4].metric("Next price", f"{row['predicted_price_1d']:.2f}")
        st.caption(f"{symbol} | latest dataset date {latest_date} | {row['signal_reason']}")

    st.plotly_chart(
        px.bar(
            signals.head(top_n),
            x="symbol",
            y="predicted_return_1d",
            color="signal",
            hover_data=["confidence", "technical_score", "signal_reason"],
            title="Next trading day signal leaderboard",
        ),
        use_container_width=True,
    )
    st.dataframe(signals.head(top_n), use_container_width=True, hide_index=True)

with history_tab:
    filtered = prices[(prices["symbol"] == symbol) & (prices["date"] >= start_timestamp)]
    st.plotly_chart(
        candlestick_with_volume(filtered, symbol, title=f"{symbol} candlestick and volume history"),
        use_container_width=True,
    )

    compare = (
        stat_forecast.rename(
            columns={
                "forecast_return_1d": "stat_return_1d",
                "forecast_price_1d": "stat_price_1d",
                "forecast_volatility_1d": "stat_volatility_1d",
            }
        )
        .merge(
            exp_forecast.rename(
                columns={
                    "forecast_return_1d": "ema_return_1d",
                    "forecast_price_1d": "ema_price_1d",
                    "forecast_volatility_1d": "ema_volatility_1d",
                }
            ),
            on=["date", "symbol", "close"],
            how="inner",
        )
    )
    st.dataframe(compare[compare["symbol"] == symbol], use_container_width=True, hide_index=True)

with portfolio_tab:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Annual return", f"{metrics['annualized_return']:.1%}")
    metric_cols[1].metric("Annual volatility", f"{metrics['annualized_volatility']:.1%}")
    metric_cols[2].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    metric_cols[3].metric("Sortino", f"{metrics['sortino_ratio']:.2f}")
    metric_cols[4].metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")

    st.plotly_chart(
        px.bar(
            selected,
            x="symbol",
            y="weight",
            color="sharpe_score",
            hover_data=["expected_return", "annualized_volatility"],
            title=f"{profile.title()} allocation",
        ),
        use_container_width=True,
    )
    st.dataframe(selected, use_container_width=True, hide_index=True)

with risk_tab:
    st.plotly_chart(
        px.scatter(
            risk.head(50),
            x="annualized_volatility",
            y="annualized_return",
            color="sharpe_ratio",
            hover_name="symbol",
            title="Risk-return map",
        ),
        use_container_width=True,
    )
    st.dataframe(risk.head(top_n), use_container_width=True, hide_index=True)

with anomaly_tab:
    anomalies = detect_market_anomalies(features)
    symbol_anomalies = anomalies[(anomalies["symbol"] == symbol) & (anomalies["date"] >= start_timestamp)]
    st.plotly_chart(
        px.scatter(
            symbol_anomalies.tail(300),
            x="date",
            y="daily_return",
            color="is_return_shock",
            size=symbol_anomalies.tail(300)["volume_zscore_20"].abs(),
            hover_data=["volume_zscore_20", "is_volume_spike"],
            title=f"{symbol} return shocks and unusual volume",
        ),
        use_container_width=True,
    )
    st.dataframe(symbol_anomalies.tail(top_n), use_container_width=True, hide_index=True)

with report_tab:
    st.plotly_chart(
        px.bar(
            report.head(top_n),
            x="symbol",
            y="portfolio_impact_score",
            color="signal",
            hover_data=["weight", "predicted_return_1d", "confidence"],
            title="Portfolio impact score",
        ),
        use_container_width=True,
    )
    st.dataframe(report.head(top_n), use_container_width=True, hide_index=True)

    st.download_button(
        label="Download CSV report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="investment_signal_report.csv",
        mime="text/csv",
    )

    summary_path = build_markdown_summary(report, metrics, profile)
    st.download_button(
        label="Download Markdown summary",
        data=summary_path.read_bytes(),
        file_name="investment_summary.md",
        mime="text/markdown",
    )
