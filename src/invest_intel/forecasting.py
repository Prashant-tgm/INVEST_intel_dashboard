from __future__ import annotations

import pandas as pd


def statistical_next_day_forecast(
    prices: pd.DataFrame,
    return_window: int = 20,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Forecast next-day return, price, and volatility with rolling statistical baselines."""
    df = prices.sort_values(["symbol", "date"]).copy()
    group = df.groupby("symbol", group_keys=False)
    df["daily_return"] = group["close"].pct_change(fill_method=None)
    df["forecast_return_1d"] = group["daily_return"].transform(lambda s: s.rolling(return_window).mean())
    df["forecast_volatility_1d"] = group["daily_return"].transform(lambda s: s.rolling(volatility_window).std())
    df["forecast_price_1d"] = df["close"] * (1 + df["forecast_return_1d"])
    latest = group.tail(1).reset_index(drop=True)
    return latest[
        [
            "date",
            "symbol",
            "close",
            "forecast_return_1d",
            "forecast_price_1d",
            "forecast_volatility_1d",
        ]
    ].sort_values("forecast_return_1d", ascending=False)


def exponential_next_day_forecast(
    prices: pd.DataFrame,
    span: int = 20,
) -> pd.DataFrame:
    """Forecast next-day return and price using exponentially weighted average returns."""
    df = prices.sort_values(["symbol", "date"]).copy()
    group = df.groupby("symbol", group_keys=False)
    df["daily_return"] = group["close"].pct_change(fill_method=None)
    df["forecast_return_1d"] = group["daily_return"].transform(lambda s: s.ewm(span=span, adjust=False).mean())
    df["forecast_volatility_1d"] = group["daily_return"].transform(lambda s: s.ewm(span=span, adjust=False).std())
    df["forecast_price_1d"] = df["close"] * (1 + df["forecast_return_1d"])
    latest = group.tail(1).reset_index(drop=True)
    return latest[
        [
            "date",
            "symbol",
            "close",
            "forecast_return_1d",
            "forecast_price_1d",
            "forecast_volatility_1d",
        ]
    ].sort_values("forecast_return_1d", ascending=False)
