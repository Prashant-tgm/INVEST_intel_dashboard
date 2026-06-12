from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "raw_dataset_dir": "dataset",
        "prices_file": "dataset/NIFTY50_all.csv",
        "metadata_file": "dataset/stock_metadata.csv",
        "processed_dir": "data/processed",
        "models_dir": "models",
        "reports_dir": "reports",
    },
    "data": {
        "date_column": "Date",
        "symbol_column": "Symbol",
        "price_column": "Close",
        "volume_column": "Volume",
        "trading_days_per_year": 252,
    },
    "features": {
        "horizons": [1, 5, 21],
        "moving_average_windows": [5, 20, 50, 100],
        "volatility_windows": [10, 20, 60],
        "rsi_window": 14,
    },
    "models": {
        "forecast_horizon_days": 5,
        "test_start_date": "2019-01-01",
        "random_state": 42,
    },
    "portfolio_profiles": {
        "conservative": {
            "max_single_stock_weight": 0.10,
            "target_volatility": 0.16,
            "risk_free_rate": 0.06,
        },
        "balanced": {
            "max_single_stock_weight": 0.15,
            "target_volatility": 0.22,
            "risk_free_rate": 0.06,
        },
        "aggressive": {
            "max_single_stock_weight": 0.25,
            "target_volatility": 0.30,
            "risk_free_rate": 0.06,
        },
    },
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML project configuration."""
    if yaml is None:
        if Path(path).resolve() == DEFAULT_CONFIG_PATH.resolve():
            return deepcopy(DEFAULT_CONFIG)
        raise ModuleNotFoundError("Install pyyaml to load custom configuration files.")

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the project root."""
    return PROJECT_ROOT.joinpath(*parts)
