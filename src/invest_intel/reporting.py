from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import project_path


def build_investment_report(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    portfolio_selection: pd.DataFrame,
) -> pd.DataFrame:
    """Combine recommendation, risk, and portfolio impact into one report table."""
    report = signals.merge(risk, on="symbol", how="left")
    weights = portfolio_selection[["symbol", "weight"]].copy() if "weight" in portfolio_selection else pd.DataFrame()
    if not weights.empty:
        report = report.merge(weights, on="symbol", how="left")
    else:
        report["weight"] = 0.0
    report["weight"] = report["weight"].fillna(0.0)
    report["portfolio_impact_score"] = (
        report["weight"] * report["predicted_return_1d"].fillna(0) * report["confidence"].fillna(0)
    )
    columns = [
        "date",
        "symbol",
        "signal",
        "confidence",
        "predicted_return_1d",
        "predicted_price_1d",
        "close",
        "weight",
        "portfolio_impact_score",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "var_95",
        "expected_shortfall_95",
        "signal_reason",
        "signal_source",
    ]
    available = [col for col in columns if col in report.columns]
    return report[available].sort_values(
        ["weight", "confidence", "predicted_return_1d"],
        ascending=[False, False, False],
    )


def save_investment_report(report: pd.DataFrame, filename: str = "investment_signal_report.csv") -> Path:
    """Save an investment report under reports."""
    out_dir = project_path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    report.to_csv(out_path, index=False)
    return out_path


def build_markdown_summary(
    report: pd.DataFrame,
    portfolio_metrics: dict[str, float],
    profile: str,
    filename: str = "investment_summary.md",
) -> Path:
    """Write a compact report summary for submission notes and PDF drafting."""
    out_dir = project_path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    top_buys = report[report["signal"].isin(["BUY", "STRONG BUY"])].head(10)
    top_sells = report[report["signal"].isin(["SELL", "STRONG SELL"])].head(10)
    portfolio_rows = report[report["weight"] > 0].sort_values("weight", ascending=False)

    lines = [
        "# NIFTY-50 Investment Intelligence Summary",
        "",
        "This report is generated only from the provided historical NIFTY-50 dataset.",
        "Signals are next-trading-day decision-support indicators, not financial advice.",
        "",
        "## Portfolio Profile",
        "",
        f"- Profile: `{profile}`",
        f"- Annualized return: {portfolio_metrics.get('annualized_return', 0):.2%}",
        f"- Annualized volatility: {portfolio_metrics.get('annualized_volatility', 0):.2%}",
        f"- Sharpe ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}",
        f"- Sortino ratio: {portfolio_metrics.get('sortino_ratio', 0):.2f}",
        f"- Maximum drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}",
        "",
        "## Portfolio Impact",
        "",
    ]

    if portfolio_rows.empty:
        lines.append("No portfolio-weighted symbols were available in the generated report.")
    else:
        for _, row in portfolio_rows.head(15).iterrows():
            lines.append(
                f"- {row['symbol']}: {row['weight']:.2%} weight, {row['signal']} signal, "
                f"{row['predicted_return_1d']:.2%} predicted next-day return, "
                f"{row['portfolio_impact_score']:.5f} impact score"
            )

    lines.extend(["", "## Top Buy Candidates", ""])
    if top_buys.empty:
        lines.append("No BUY or STRONG BUY signals were generated.")
    else:
        for _, row in top_buys.iterrows():
            lines.append(
                f"- {row['symbol']}: {row['signal']} at {row['confidence']:.0%} confidence; "
                f"next-day return {row['predicted_return_1d']:.2%}; {row['signal_reason']}"
            )

    lines.extend(["", "## Top Sell / Avoid Candidates", ""])
    if top_sells.empty:
        lines.append("No SELL or STRONG SELL signals were generated.")
    else:
        for _, row in top_sells.iterrows():
            lines.append(
                f"- {row['symbol']}: {row['signal']} at {row['confidence']:.0%} confidence; "
                f"next-day return {row['predicted_return_1d']:.2%}; {row['signal_reason']}"
            )

    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Technical indicators include moving averages, EMA, RSI, MACD, Bollinger Bands, volatility, momentum, and volume z-score.",
            "- Forecasting uses time-based validation to reduce look-ahead bias.",
            "- Risk analytics include volatility, Sharpe ratio, Sortino ratio, maximum drawdown, VaR, and expected shortfall.",
            "- Portfolio impact score combines allocation weight, predicted next-day return, and signal confidence.",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
