from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_with_volume(
    prices: pd.DataFrame,
    symbol: str,
    title: str | None = None,
) -> go.Figure:
    """Build a candlestick chart with volume bars for historical stock inspection."""
    df = prices[prices["symbol"] == symbol].sort_values("date").copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], name="Volume", marker_color="#6b7280"),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title or f"{symbol} candlestick history",
        xaxis_rangeslider_visible=False,
        height=620,
        margin=dict(l=20, r=20, t=50, b=20),
        legend_orientation="h",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig
