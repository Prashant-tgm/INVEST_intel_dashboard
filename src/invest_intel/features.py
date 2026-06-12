from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_features(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21),
) -> pd.DataFrame:
    """Add leakage-safe per-symbol technical indicators and targets."""
    df = prices.sort_values(["symbol", "date"]).copy()
    group = df.groupby("symbol", group_keys=False)

    df["daily_return"] = group["close"].pct_change(fill_method=None)
    df["log_return"] = np.log1p(df["daily_return"])
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"]
    df["close_to_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["gap_return"] = (df["open"] - df["prev_close"]) / df["prev_close"]

    for window in (5, 20, 50, 100):
        df[f"ma_{window}"] = group["close"].transform(lambda s: s.rolling(window).mean())
        df[f"close_to_ma_{window}"] = df["close"] / df[f"ma_{window}"] - 1

    for window in (10, 20, 60):
        df[f"volatility_{window}"] = group["daily_return"].transform(lambda s: s.rolling(window).std())
        df[f"momentum_{window}"] = group["close"].pct_change(window, fill_method=None)

    df["ema_12"] = group["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df["ema_26"] = group["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = group["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["rsi_14"] = group["close"].transform(_rsi)

    rolling_mean_20 = group["close"].transform(lambda s: s.rolling(20).mean())
    rolling_std_20 = group["close"].transform(lambda s: s.rolling(20).std())
    df["bollinger_upper_20"] = rolling_mean_20 + 2 * rolling_std_20
    df["bollinger_lower_20"] = rolling_mean_20 - 2 * rolling_std_20
    df["bollinger_width_20"] = (df["bollinger_upper_20"] - df["bollinger_lower_20"]) / rolling_mean_20

    df["volume_zscore_20"] = group["volume"].transform(
        lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std()
    )

    for horizon in horizons:
        future_col = f"future_return_{horizon}d"
        target_col = f"target_direction_{horizon}d"
        df[future_col] = group["close"].transform(lambda s: s.shift(-horizon) / s - 1)
        df[target_col] = np.where(df[future_col].isna(), np.nan, (df[future_col] > 0).astype("int"))

    return df


def modeling_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns for predictive models."""
    excluded_prefixes = ("future_return_", "target_direction_")
    excluded = {"date", "symbol", "series", "company_name", "industry", "isin_code"}
    return [
        col
        for col in df.columns
        if col not in excluded
        and not col.startswith(excluded_prefixes)
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def drop_modeling_na(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Drop rows that cannot be used by a supervised model."""
    required = ["date", "symbol", target_col, *feature_cols]
    return df.dropna(subset=required).reset_index(drop=True)
