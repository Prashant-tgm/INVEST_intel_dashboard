from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .config import project_path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


@dataclass(frozen=True)
class SignalThresholds:
    """Thresholds for translating model and indicator output into action labels."""

    buy_return_threshold: float = 0.004
    sell_return_threshold: float = -0.004
    strong_buy_return_threshold: float = 0.012
    strong_sell_return_threshold: float = -0.012


def _safe_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    return default if pd.isna(value) else float(value)


def technical_signal_score(row: pd.Series) -> tuple[float, list[str]]:
    """Create a compact technical score from MA, EMA, RSI, MACD, Bollinger, volume, and momentum."""
    score = 0.0
    reasons: list[str] = []

    close_to_ma_20 = _safe_value(row, "close_to_ma_20")
    close_to_ma_50 = _safe_value(row, "close_to_ma_50")
    rsi = _safe_value(row, "rsi_14", 50.0)
    macd = _safe_value(row, "macd")
    macd_signal = _safe_value(row, "macd_signal")
    momentum_20 = _safe_value(row, "momentum_20")
    volatility_20 = _safe_value(row, "volatility_20")
    volume_zscore = _safe_value(row, "volume_zscore_20")
    close = _safe_value(row, "close")
    bollinger_upper = _safe_value(row, "bollinger_upper_20", np.nan)
    bollinger_lower = _safe_value(row, "bollinger_lower_20", np.nan)

    if close_to_ma_20 > 0:
        score += 0.8
        reasons.append("close above 20-day MA")
    else:
        score -= 0.8
        reasons.append("close below 20-day MA")

    if close_to_ma_50 > 0:
        score += 0.7
        reasons.append("close above 50-day MA")
    else:
        score -= 0.7
        reasons.append("close below 50-day MA")

    if macd > macd_signal:
        score += 0.8
        reasons.append("MACD bullish")
    else:
        score -= 0.8
        reasons.append("MACD bearish")

    if rsi < 30:
        score += 0.7
        reasons.append("RSI oversold")
    elif rsi > 70:
        score -= 0.7
        reasons.append("RSI overbought")
    else:
        reasons.append("RSI neutral")

    if momentum_20 > 0:
        score += 0.5
        reasons.append("positive 20-day momentum")
    elif momentum_20 < 0:
        score -= 0.5
        reasons.append("negative 20-day momentum")

    if not pd.isna(bollinger_lower) and close <= bollinger_lower:
        score += 0.5
        reasons.append("near lower Bollinger band")
    elif not pd.isna(bollinger_upper) and close >= bollinger_upper:
        score -= 0.5
        reasons.append("near upper Bollinger band")

    if volatility_20 > 0.035:
        score -= 0.3
        reasons.append("high short-term volatility")

    if volume_zscore > 2:
        reasons.append("unusual volume spike")

    return float(score), reasons[:5]


def signal_from_prediction(
    predicted_return: float,
    technical_score: float,
    thresholds: SignalThresholds | None = None,
) -> tuple[str, float]:
    """Convert next-day predicted return and indicator score into an action label."""
    limits = thresholds or SignalThresholds()
    blended_score = predicted_return + (technical_score * 0.002)

    if blended_score >= limits.strong_buy_return_threshold:
        label = "STRONG BUY"
    elif blended_score >= limits.buy_return_threshold:
        label = "BUY"
    elif blended_score <= limits.strong_sell_return_threshold:
        label = "STRONG SELL"
    elif blended_score <= limits.sell_return_threshold:
        label = "SELL"
    else:
        label = "HOLD"

    confidence = min(0.99, max(0.50, 0.50 + abs(blended_score) * 18 + abs(technical_score) * 0.04))
    return label, float(confidence)


def latest_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Return the latest available feature row for every symbol."""
    return (
        features.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def indicator_based_next_day_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Create next-trading-day signals using technical indicators when no trained model is available."""
    latest = latest_feature_rows(features)
    records = []
    for _, row in latest.iterrows():
        score, reasons = technical_signal_score(row)
        predicted_return = score * 0.003
        action, confidence = signal_from_prediction(predicted_return, score)
        records.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "close": row["close"],
                "predicted_return_1d": predicted_return,
                "predicted_price_1d": row["close"] * (1 + predicted_return),
                "technical_score": score,
                "signal": action,
                "confidence": confidence,
                "signal_reason": "; ".join(reasons),
                "signal_source": "technical_indicators",
            }
        )
    return pd.DataFrame(records).sort_values(["signal", "confidence"], ascending=[True, False])


def model_based_next_day_signals(
    features: pd.DataFrame,
    feature_cols: list[str],
    estimator: object,
) -> pd.DataFrame:
    """Create next-day signals from a trained one-day return model plus technical confirmation."""
    latest = latest_feature_rows(features).dropna(subset=feature_cols).copy()
    if latest.empty:
        return indicator_based_next_day_signals(features)

    predicted = estimator.predict(latest[feature_cols])
    records = []
    for idx, (_, row) in enumerate(latest.iterrows()):
        score, reasons = technical_signal_score(row)
        predicted_return = float(predicted[idx])
        action, confidence = signal_from_prediction(predicted_return, score)
        records.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "close": row["close"],
                "predicted_return_1d": predicted_return,
                "predicted_price_1d": row["close"] * (1 + predicted_return),
                "technical_score": score,
                "signal": action,
                "confidence": confidence,
                "signal_reason": "; ".join(reasons),
                "signal_source": "model_plus_indicators",
            }
        )
    return pd.DataFrame(records).sort_values("confidence", ascending=False)


def signals_from_saved_model(
    features: pd.DataFrame,
    artifact_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a saved one-day model artifact if available, otherwise use technical signals.

    Some locked-down Windows environments block compiled sklearn/scipy DLLs.
    In that case the dashboard still runs with indicator-only signals.
    """
    path = Path(artifact_path) if artifact_path else project_path("models", "stock_return_1d_model.joblib")
    if not path.exists():
        return indicator_based_next_day_signals(features)

    try:
        import joblib

        artifact = joblib.load(path)
        return model_based_next_day_signals(features, artifact["feature_cols"], artifact["model"])
    except Exception:
        return indicator_based_next_day_signals(features)
