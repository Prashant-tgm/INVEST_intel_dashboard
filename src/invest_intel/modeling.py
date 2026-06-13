from __future__ import annotations

from dataclasses import dataclass
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    estimator: object
    metrics: dict[str, float]
    predictions: pd.DataFrame


class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a probability classifier and apply a validation-tuned decision threshold."""

    def __init__(self, estimator: object, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ThresholdedClassifier":
        self.estimator.fit(x, y)
        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(x)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)[:, 1]
        return (proba >= self.threshold).astype(int)


class AveragingClassifier(BaseEstimator, ClassifierMixin):
    """Average probabilities from heterogeneous classifiers."""

    def __init__(self, estimators: list[tuple[str, object]], weights: list[float] | None = None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "AveragingClassifier":
        self.fitted_estimators_ = []
        for name, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(x, y)
            self.fitted_estimators_.append((name, fitted))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probabilities = np.stack(
            [estimator.predict_proba(x) for _, estimator in self.fitted_estimators_],
            axis=0,
        )
        return np.average(probabilities, axis=0, weights=self.weights)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


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


def evaluate_direction_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate binary up/down predictions with accuracy and class-balance aware metrics."""
    metrics = {
        "directional_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def _validation_split(
    train: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(train["date"].dropna().unique()))
    if len(unique_dates) < 10:
        return train, train.iloc[0:0].copy()
    split_idx = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * (1 - validation_fraction))))
    validation_start = unique_dates[split_idx]
    train_core = train[train["date"] < validation_start].copy()
    validation = train[train["date"] >= validation_start].copy()
    return train_core, validation


def _best_probability_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.35, 0.65, 61)
    scores = [accuracy_score(y_true, y_prob >= threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _direction_estimators(random_state: int = 42) -> dict[str, object]:
    estimators: dict[str, object] = {
        "logistic_regression_direction": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.7,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_direction": HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            class_weight="balanced",
            random_state=random_state,
        ),
        "extra_trees_direction": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        estimators["lightgbm_direction"] = LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.025,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError:
        pass

    return estimators


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
    """Train direction classifiers and return the best validation-selected model."""
    results = train_direction_models(
        df,
        feature_cols,
        target_col=target_col,
        test_start_date=test_start_date,
        random_state=random_state,
    )
    return max(
        results,
        key=lambda result: (
            result.metrics["directional_accuracy"],
            result.metrics["balanced_accuracy"],
        ),
    )


def train_direction_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_direction_5d",
    test_start_date: str = "2019-01-01",
    random_state: int = 42,
) -> list[ModelResult]:
    """Train stronger direction classifiers with validation-tuned probability thresholds."""
    train, test = time_based_split(df, test_start_date)
    train_core, validation = _validation_split(train)
    results = []
    estimators = _direction_estimators(random_state)
    validation_probabilities: dict[str, np.ndarray] = {}
    validation_scores: dict[str, float] = {}

    for name, estimator in estimators.items():
        validation_estimator = clone(estimator)
        validation_estimator.fit(train_core[feature_cols], train_core[target_col])

        threshold = 0.5
        validation_accuracy = np.nan
        if not validation.empty:
            validation_prob = validation_estimator.predict_proba(validation[feature_cols])[:, 1]
            validation_probabilities[name] = validation_prob
            threshold = _best_probability_threshold(validation[target_col], validation_prob)
            validation_accuracy = accuracy_score(validation[target_col], validation_prob >= threshold)
            validation_scores[name] = float(validation_accuracy)

        final_estimator = ThresholdedClassifier(clone(estimator), threshold=threshold)
        final_estimator.fit(train[feature_cols], train[target_col])

        prob = final_estimator.predict_proba(test[feature_cols])[:, 1]
        pred = (prob >= threshold).astype(int)
        predictions = test[["date", "symbol", target_col]].copy()
        predictions["prediction"] = pred
        predictions["up_probability"] = prob

        metrics = evaluate_direction_predictions(test[target_col], pred, prob)
        metrics["decision_threshold"] = float(threshold)
        metrics["validation_directional_accuracy"] = float(validation_accuracy)
        results.append(
            ModelResult(
                name=name,
                estimator=final_estimator,
                metrics=metrics,
                predictions=predictions,
            )
        )

    if len(estimators) > 1 and not validation.empty:
        names = list(validation_probabilities)
        ensemble_prob = np.average(
            np.column_stack([validation_probabilities[name] for name in names]),
            axis=1,
            weights=[max(validation_scores[name] - 0.5, 0.001) for name in names],
        )
        threshold = _best_probability_threshold(validation[target_col], ensemble_prob)
        ensemble = ThresholdedClassifier(
            AveragingClassifier(
                [(name, estimators[name]) for name in names],
                weights=[max(validation_scores[name] - 0.5, 0.001) for name in names],
            ),
            threshold=threshold,
        )
        ensemble.fit(train[feature_cols], train[target_col])
        prob = ensemble.predict_proba(test[feature_cols])[:, 1]
        pred = (prob >= threshold).astype(int)
        predictions = test[["date", "symbol", target_col]].copy()
        predictions["prediction"] = pred
        predictions["up_probability"] = prob
        metrics = evaluate_direction_predictions(test[target_col], pred, prob)
        metrics["decision_threshold"] = float(threshold)
        metrics["validation_directional_accuracy"] = float(accuracy_score(validation[target_col], ensemble_prob >= threshold))
        results.append(
            ModelResult(
                name="soft_voting_direction",
                estimator=ensemble,
                metrics=metrics,
                predictions=predictions,
            )
        )
    return results


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
                    hidden_layer_sizes=(256, 128, 64, 32),
                    activation="relu",
                    alpha=0.0005,
                    learning_rate_init=0.0007,
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=400,
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
