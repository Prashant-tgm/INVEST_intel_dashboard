from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from invest_intel.data import attach_metadata, build_returns_matrix, load_prices, save_processed
from invest_intel.features import add_technical_features, drop_modeling_na, modeling_feature_columns
from invest_intel.modeling import train_direction_models, train_return_models
from invest_intel.portfolio import construct_profile_portfolio
from invest_intel.reporting import build_investment_report, build_markdown_summary, save_investment_report
from invest_intel.risk import risk_summary
from invest_intel.signals import signals_from_saved_model


def main() -> None:
    profile = "balanced"
    print("Loading prices and metadata...")
    prices = attach_metadata(load_prices())

    print("Building technical features...")
    features = add_technical_features(prices)
    feature_path = save_processed(features, "nifty50_features.csv")
    print(f"Saved features: {feature_path}")

    print("Training next-day return models...")
    feature_cols = modeling_feature_columns(features)
    return_df = drop_modeling_na(features, "future_return_1d", feature_cols)
    return_results = train_return_models(
        return_df,
        feature_cols,
        target_col="future_return_1d",
        test_start_date="2019-01-01",
    )
    metrics = pd.DataFrame([{"model": result.name, **result.metrics} for result in return_results])
    best = max(return_results, key=lambda result: result.metrics["directional_accuracy"])
    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    joblib.dump(
        {"model": best.estimator, "feature_cols": feature_cols, "target": "future_return_1d"},
        models_dir / "stock_return_1d_model.joblib",
    )
    metrics.to_csv(reports_dir / "model_metrics_1d.csv", index=False)
    print(metrics.to_string(index=False))

    print("Training next-day direction model...")
    direction_df = drop_modeling_na(features, "target_direction_1d", feature_cols)
    direction_results = train_direction_models(
        direction_df,
        feature_cols,
        target_col="target_direction_1d",
        test_start_date="2019-01-01",
    )
    direction_metrics = pd.DataFrame([{"model": result.name, **result.metrics} for result in direction_results])
    direction_result = max(
        direction_results,
        key=lambda result: (
            result.metrics["directional_accuracy"],
            result.metrics["balanced_accuracy"],
        ),
    )
    joblib.dump(
        {"model": direction_result.estimator, "feature_cols": feature_cols, "target": "target_direction_1d"},
        models_dir / "stock_direction_1d_model.joblib",
    )
    direction_metrics.to_csv(reports_dir / "direction_model_metrics_1d.csv", index=False)
    print(direction_metrics.to_string(index=False))

    print("Building portfolio, risk, signals, and reports...")
    recent_prices = prices[prices["date"] >= "2015-01-01"]
    returns = build_returns_matrix(recent_prices, min_obs=252).dropna(how="all")
    risk = risk_summary(returns)
    weights, portfolio_metrics, selected = construct_profile_portfolio(returns, profile=profile)
    signals = signals_from_saved_model(features)
    report = build_investment_report(signals, risk, selected)
    csv_path = save_investment_report(report)
    md_path = build_markdown_summary(report, portfolio_metrics, profile)

    print(f"Saved investment report: {csv_path}")
    print(f"Saved markdown summary: {md_path}")
    print(f"Portfolio weights sum: {weights.sum():.6f}")


if __name__ == "__main__":
    main()
