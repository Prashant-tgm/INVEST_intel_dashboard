from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from invest_intel.data import attach_metadata, build_returns_matrix, load_prices
from invest_intel.features import add_technical_features
from invest_intel.forecasting import exponential_next_day_forecast, statistical_next_day_forecast
from invest_intel.portfolio import construct_profile_portfolio
from invest_intel.reporting import build_investment_report
from invest_intel.risk import risk_summary
from invest_intel.signals import signals_from_saved_model
from invest_intel.visualization import candlestick_with_volume


def main() -> None:
    prices = attach_metadata(load_prices())
    subset = prices[prices["symbol"].isin(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])].copy()
    features = add_technical_features(subset)
    signals = signals_from_saved_model(features)
    returns = build_returns_matrix(subset[subset["date"] >= "2015-01-01"], min_obs=126).dropna(how="all")
    risk = risk_summary(returns)
    weights, metrics, selected = construct_profile_portfolio(returns, profile="balanced", top_n=3)
    report = build_investment_report(signals, risk, selected)
    stat = statistical_next_day_forecast(subset)
    ema = exponential_next_day_forecast(subset)
    fig = candlestick_with_volume(subset[subset["date"] >= "2020-01-01"], "RELIANCE")

    assert not signals.empty
    assert not risk.empty
    assert not report.empty
    assert not stat.empty
    assert not ema.empty
    assert len(fig.data) == 2
    assert round(float(weights.sum()), 6) == 1.0
    assert "sharpe_ratio" in metrics
    print("smoke_test_ok")


if __name__ == "__main__":
    main()
