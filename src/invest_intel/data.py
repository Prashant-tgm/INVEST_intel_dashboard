from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, project_path


RAW_COLUMN_MAP = {
    "Date": "date",
    "Symbol": "symbol",
    "Series": "series",
    "Prev Close": "prev_close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Last": "last",
    "Close": "close",
    "VWAP": "vwap",
    "Volume": "volume",
    "Turnover": "turnover",
    "Trades": "trades",
    "Deliverable Volume": "deliverable_volume",
    "%Deliverble": "deliverable_pct",
}


def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    """Load and normalize the combined NIFTY-50 price history."""
    cfg = load_config()
    raw_path = Path(path) if path else project_path(cfg["paths"]["prices_file"])
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    df = df.rename(columns=RAW_COLUMN_MAP)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    numeric_cols = [
        "prev_close",
        "open",
        "high",
        "low",
        "last",
        "close",
        "vwap",
        "volume",
        "turnover",
        "trades",
        "deliverable_volume",
        "deliverable_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Load stock metadata and normalize field names."""
    cfg = load_config()
    raw_path = Path(path) if path else project_path(cfg["paths"]["metadata_file"])
    meta = pd.read_csv(raw_path)
    return meta.rename(
        columns={
            "Company Name": "company_name",
            "Industry": "industry",
            "Symbol": "symbol",
            "Series": "series",
            "ISIN Code": "isin_code",
        }
    )


def attach_metadata(prices: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach company metadata where symbol-level metadata exists."""
    meta = metadata if metadata is not None else load_metadata()
    return prices.merge(
        meta[["symbol", "company_name", "industry", "isin_code"]],
        on="symbol",
        how="left",
    )


def build_returns_matrix(
    prices: pd.DataFrame,
    price_col: str = "close",
    min_obs: int = 252,
) -> pd.DataFrame:
    """Create a daily returns matrix with symbols as columns."""
    pivot = prices.pivot(index="date", columns="symbol", values=price_col).sort_index()
    returns = pivot.pct_change(fill_method=None)
    returns = returns.dropna(axis=1, thresh=min_obs)
    return returns


def dataset_audit(dataset_dir: str | Path | None = None) -> dict[str, object]:
    """Return a compact audit summary for reporting and notebook checks."""
    cfg = load_config()
    root = Path(dataset_dir) if dataset_dir else project_path(cfg["paths"]["raw_dataset_dir"])
    prices = load_prices(root / "NIFTY50_all.csv")
    metadata = load_metadata(root / "stock_metadata.csv")

    ticker_files = sorted(root.glob("*.csv"))
    tiny_files = [file.name for file in ticker_files if file.name != "stock_metadata.csv" and file.stat().st_size < 1024]
    missing_counts = prices.isna().sum().sort_values(ascending=False).to_dict()
    metadata_symbols = set(metadata["symbol"])
    price_symbols = set(prices["symbol"])

    return {
        "rows": int(len(prices)),
        "columns": list(prices.columns),
        "date_min": str(prices["date"].min().date()),
        "date_max": str(prices["date"].max().date()),
        "price_symbol_count": int(prices["symbol"].nunique()),
        "metadata_symbol_count": int(metadata["symbol"].nunique()),
        "metadata_industry_count": int(metadata["industry"].nunique()),
        "symbols_without_metadata": sorted(price_symbols - metadata_symbols),
        "metadata_without_prices": sorted(metadata_symbols - price_symbols),
        "tiny_csv_files": tiny_files,
        "top_missing": dict(list(missing_counts.items())[:8]),
    }


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save a processed dataframe under data/processed."""
    cfg = load_config()
    out_dir = project_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def load_processed(filename: str) -> pd.DataFrame:
    """Load a processed dataframe from data/processed."""
    cfg = load_config()
    path = project_path(cfg["paths"]["processed_dir"], filename)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["date"])
