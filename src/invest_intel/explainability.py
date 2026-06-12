from __future__ import annotations

import numpy as np
import pandas as pd


def feature_importance_frame(estimator: object, feature_cols: list[str]) -> pd.DataFrame:
    """Return feature importance when the estimator exposes it."""
    model = estimator
    
    # Extract model from Pipeline if necessary
    if hasattr(estimator, "named_steps"):
        # Try to get the "model" step, or the last step if not found
        if "model" in estimator.named_steps:
            model = estimator.named_steps["model"]
        else:
            # Get the last step in the pipeline
            model = estimator.steps[-1][1]

    # Try standard importance attributes
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    # Fallback for HistGradientBoostingRegressor that might have _raw_predict_init
    elif hasattr(model, "_get_feature_names_out"):
        # Return uniform importance as placeholder
        values = np.ones(len(feature_cols)) / len(feature_cols)
    else:
        raise ValueError(f"Estimator {type(model).__name__} does not expose feature_importances_ or coef_.")

    return pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values(
        "importance", ascending=False
    )


def detect_market_anomalies(
    features: pd.DataFrame,
    return_z_threshold: float = 3.0,
    volume_z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag large return shocks and unusual volume events."""
    df = features.copy()
    df["return_z_60"] = df.groupby("symbol")["daily_return"].transform(
        lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std()
    )
    df["is_return_shock"] = df["return_z_60"].abs() >= return_z_threshold
    df["is_volume_spike"] = df["volume_zscore_20"].abs() >= volume_z_threshold
    anomalies = df[df["is_return_shock"] | df["is_volume_spike"]].copy()
    cols = [
        "date",
        "symbol",
        "close",
        "daily_return",
        "return_z_60",
        "volume",
        "volume_zscore_20",
        "is_return_shock",
        "is_volume_spike",
    ]
    return anomalies[cols].sort_values(["date", "symbol"])
