from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def max_drawdown(return_series: pd.Series) -> float:
    cumulative = (1 + return_series.dropna()).cumprod()
    if cumulative.empty:
        return np.nan
    drawdown = cumulative / cumulative.cummax() - 1
    return float(drawdown.min())


def annualized_return(return_series: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = return_series.dropna()
    if returns.empty:
        return np.nan
    cumulative = float((1 + returns).prod())
    years = len(returns) / periods_per_year
    return cumulative ** (1 / years) - 1 if years > 0 else np.nan


def annualized_volatility(return_series: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(return_series.dropna().std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    return_series: pd.Series,
    risk_free_rate: float = 0.06,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    ann_return = annualized_return(return_series, periods_per_year)
    ann_vol = annualized_volatility(return_series, periods_per_year)
    return (ann_return - risk_free_rate) / ann_vol if ann_vol and ann_vol > 0 else np.nan


def sortino_ratio(
    return_series: pd.Series,
    risk_free_rate: float = 0.06,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    returns = return_series.dropna()
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    ann_return = annualized_return(returns, periods_per_year)
    return (ann_return - risk_free_rate) / downside if downside and downside > 0 else np.nan


def value_at_risk(return_series: pd.Series, confidence: float = 0.95) -> float:
    returns = return_series.dropna()
    return float(np.quantile(returns, 1 - confidence)) if not returns.empty else np.nan


def expected_shortfall(return_series: pd.Series, confidence: float = 0.95) -> float:
    returns = return_series.dropna()
    if returns.empty:
        return np.nan
    var = value_at_risk(returns, confidence)
    return float(returns[returns <= var].mean())


def risk_summary(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.06,
) -> pd.DataFrame:
    """Compute symbol-level risk metrics from a daily returns matrix."""
    records = []
    for symbol in returns.columns:
        series = returns[symbol]
        records.append(
            {
                "symbol": symbol,
                "annualized_return": annualized_return(series),
                "annualized_volatility": annualized_volatility(series),
                "sharpe_ratio": sharpe_ratio(series, risk_free_rate),
                "sortino_ratio": sortino_ratio(series, risk_free_rate),
                "max_drawdown": max_drawdown(series),
                "var_95": value_at_risk(series, 0.95),
                "expected_shortfall_95": expected_shortfall(series, 0.95),
            }
        )
    return pd.DataFrame(records).sort_values("sharpe_ratio", ascending=False)


def portfolio_return_series(returns: pd.DataFrame, weights: pd.Series | dict[str, float]) -> pd.Series:
    weights_series = pd.Series(weights, dtype=float)
    aligned = returns[weights_series.index].dropna(how="all")
    normalized_weights = weights_series / weights_series.sum()
    return aligned.fillna(0).dot(normalized_weights)


def portfolio_metrics(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float],
    risk_free_rate: float = 0.06,
) -> dict[str, float]:
    series = portfolio_return_series(returns, weights)
    return {
        "annualized_return": annualized_return(series),
        "annualized_volatility": annualized_volatility(series),
        "sharpe_ratio": sharpe_ratio(series, risk_free_rate),
        "sortino_ratio": sortino_ratio(series, risk_free_rate),
        "max_drawdown": max_drawdown(series),
        "var_95": value_at_risk(series, 0.95),
        "expected_shortfall_95": expected_shortfall(series, 0.95),
    }
