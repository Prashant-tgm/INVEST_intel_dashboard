from __future__ import annotations

from dataclasses import dataclass
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    estimator: object
    metrics: dict[str, float]
    predictions: pd.DataFrame


def time_based_split(
    df: pd.DataFrame,
    test_start_date: str = "2019-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = pd.Timestamp(test_start_date)
    train = df[df["date"] < test_start].copy()
    test = df[df["date"] >= test_start].copy()
    return train, test


def evaluate_return_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    direction_accuracy = accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(direction_accuracy),
    }


def train_return_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "future_return_5d",
    test_start_date: str = "2019-01-01",
    random_state: int = 42,
) -> list[ModelResult]:
    """Train baseline models for future return prediction."""
    train, test = time_based_split(df, test_start_date)
    x_train, y_train = train[feature_cols], train[target_col]
    x_test, y_test = test[feature_cols], test[target_col]

    models = {
        "ridge_regression": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=random_state))]
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
    }

    results = []
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        predictions = test[["date", "symbol", target_col]].copy()
        predictions["prediction"] = pred
        results.append(
            ModelResult(
                name=name,
                estimator=estimator,
                metrics=evaluate_return_predictions(y_test, pred),
                predictions=predictions,
            )
        )
    return results


def train_direction_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_direction_5d",
    test_start_date: str = "2019-01-01",
    random_state: int = 42,
) -> ModelResult:
    """Train a simple direction classifier for upward/downward movement."""
    train, test = time_based_split(df, test_start_date)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=random_state,
    )
    model.fit(train[feature_cols], train[target_col])
    pred = model.predict(test[feature_cols])
    prob = model.predict_proba(test[feature_cols])[:, 1]
    predictions = test[["date", "symbol", target_col]].copy()
    predictions["prediction"] = pred
    predictions["up_probability"] = prob
    return ModelResult(
        name="random_forest_direction",
        estimator=model,
        metrics={"directional_accuracy": float(accuracy_score(test[target_col], pred))},
        predictions=predictions,
    )


def train_neural_network_return_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "future_return_1d",
    test_start_date: str = "2019-01-01",
    random_state: int = 42,
) -> ModelResult:
    """Train a lightweight neural-network baseline for return forecasting."""
    train, test = time_based_split(df, test_start_date)
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    max_iter=200,
                    random_state=random_state,
                ),
            ),
        ]
    )
    estimator.fit(train[feature_cols], train[target_col])
    pred = estimator.predict(test[feature_cols])
    predictions = test[["date", "symbol", target_col]].copy()
    predictions["prediction"] = pred
    return ModelResult(
        name="mlp_neural_network_return",
        estimator=estimator,
        metrics=evaluate_return_predictions(test[target_col], pred),
        predictions=predictions,
    )
