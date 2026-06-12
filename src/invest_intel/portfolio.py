from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import portfolio_metrics


PROFILE_SETTINGS = {
    "conservative": {"max_single_stock_weight": 0.10, "risk_free_rate": 0.06},
    "balanced": {"max_single_stock_weight": 0.15, "risk_free_rate": 0.06},
    "aggressive": {"max_single_stock_weight": 0.25, "risk_free_rate": 0.06},
}


def _cap_and_normalize(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights.clip(lower=0)
    if weights.sum() == 0:
        weights[:] = 1 / len(weights)
    weights = weights / weights.sum()

    for _ in range(20):
        excess = weights[weights > max_weight] - max_weight
        if excess.empty:
            break
        weights[weights > max_weight] = max_weight
        redistribute_to = weights[weights < max_weight].index
        if len(redistribute_to) == 0:
            break
        weights.loc[redistribute_to] += excess.sum() * weights.loc[redistribute_to] / weights.loc[redistribute_to].sum()
    return weights / weights.sum()


def score_assets(
    returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    risk_free_rate: float = 0.06,
) -> pd.DataFrame:
    """Rank assets by simple risk-adjusted statistics."""
    expected = expected_returns if expected_returns is not None else returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe = (expected - risk_free_rate) / volatility.replace(0, np.nan)
    return pd.DataFrame(
        {
            "symbol": returns.columns,
            "expected_return": expected.reindex(returns.columns).values,
            "annualized_volatility": volatility.reindex(returns.columns).values,
            "sharpe_score": sharpe.reindex(returns.columns).values,
        }
    ).sort_values("sharpe_score", ascending=False)


def construct_profile_portfolio(
    returns: pd.DataFrame,
    profile: str = "balanced",
    top_n: int = 12,
    expected_returns: pd.Series | None = None,
) -> tuple[pd.Series, dict[str, float], pd.DataFrame]:
    """Construct a transparent profile portfolio using risk-adjusted scores."""
    if profile not in PROFILE_SETTINGS:
        raise ValueError(f"Unknown profile: {profile}")

    settings = PROFILE_SETTINGS[profile]
    scores = score_assets(returns, expected_returns, settings["risk_free_rate"]).dropna()
    selected = scores.head(top_n).copy()

    if profile == "conservative":
        raw = 1 / selected["annualized_volatility"].replace(0, np.nan)
    elif profile == "aggressive":
        raw = selected["sharpe_score"].clip(lower=0) ** 1.5
    else:
        raw = selected["sharpe_score"].clip(lower=0)

    raw.index = selected["symbol"]
    weights = _cap_and_normalize(raw, settings["max_single_stock_weight"])
    metrics = portfolio_metrics(returns, weights, settings["risk_free_rate"])
    selected["weight"] = selected["symbol"].map(weights)
    return weights.sort_values(ascending=False), metrics, selected.sort_values("weight", ascending=False)


def random_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int = 2000,
    risk_free_rate: float = 0.06,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate random long-only portfolios for an efficient-frontier style plot."""
    rng = np.random.default_rng(random_state)
    records = []
    symbols = list(returns.columns)
    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252

    for _ in range(n_portfolios):
        weights = rng.random(len(symbols))
        weights = weights / weights.sum()
        expected_return = float(np.dot(weights, mean_returns))
        volatility = float(np.sqrt(weights.T @ cov.values @ weights))
        sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else np.nan
        records.append(
            {
                "expected_return": expected_return,
                "annualized_volatility": volatility,
                "sharpe_ratio": sharpe,
                "weights": dict(zip(symbols, weights)),
            }
        )
    return pd.DataFrame(records)
