from __future__ import annotations

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _unwrap_estimator(estimator: object) -> object:
    model = estimator
    if hasattr(model, "estimator"):
        model = model.estimator
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model.steps[-1][1])
    return model


def feature_importance_frame(estimator: object, feature_cols: list[str]) -> pd.DataFrame:
    """Return native feature importance when an estimator exposes it."""
    model = _unwrap_estimator(estimator)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        raise ValueError(
            f"Estimator {type(model).__name__} does not expose native feature importance. "
            "Use permutation_importance_frame or shap_feature_importance_frame instead."
        )

    return pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values(
        "importance", ascending=False
    )


def permutation_importance_frame(
    estimator: object,
    x: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    scoring: str = "accuracy",
    n_repeats: int = 5,
    max_rows: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate model-agnostic feature importance on a bounded validation sample."""
    if len(x) > max_rows:
        x_eval = x.sample(max_rows, random_state=random_state)
        y_eval = y.loc[x_eval.index]
    else:
        x_eval = x
        y_eval = y

    result = permutation_importance(
        estimator,
        x_eval[feature_cols],
        y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)


def shap_feature_importance_frame(
    estimator: object,
    x: pd.DataFrame,
    feature_cols: list[str],
    max_background: int = 200,
    max_rows: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values for tree-style estimators when SHAP is installed."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap to compute SHAP feature importance.") from exc

    x_features = x[feature_cols]
    background = x_features.sample(min(max_background, len(x_features)), random_state=random_state)
    x_eval = x_features.sample(min(max_rows, len(x_features)), random_state=random_state + 1)

    model = estimator
    if hasattr(model, "estimator"):
        model = model.estimator

    if hasattr(model, "named_steps"):
        steps = model.steps
        for _, transformer in steps[:-1]:
            background = pd.DataFrame(
                transformer.transform(background),
                columns=feature_cols,
                index=background.index,
            )
            x_eval = pd.DataFrame(
                transformer.transform(x_eval),
                columns=feature_cols,
                index=x_eval.index,
            )
        model = steps[-1][1]

    explainer = shap.Explainer(model, background)
    values = explainer(x_eval).values
    if isinstance(values, list):
        values = values[-1]
    if values.ndim == 3:
        values = values[:, :, -1]

    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values(
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
