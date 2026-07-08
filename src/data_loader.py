"""Fetch and persist financial data from YFinance."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ("TSLA", "BND", "SPY")
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-06-30"


def fetch_asset_data(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker."""
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Ticker"] = ticker
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def fetch_all_assets(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers."""
    return {ticker: fetch_asset_data(ticker, start, end) for ticker in tickers}


def combine_close_prices(asset_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a wide DataFrame of adjusted close prices indexed by date."""
    frames = []
    for ticker, df in asset_data.items():
        subset = df[["Date", "Adj Close"]].copy()
        subset = subset.rename(columns={"Adj Close": ticker})
        frames.append(subset.set_index("Date"))

    combined = pd.concat(frames, axis=1).sort_index()
    combined.index.name = "Date"
    return combined


def save_processed_data(
    asset_data: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> Path:
    """Save individual ticker CSVs and combined close prices."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ticker, df in asset_data.items():
        df.to_csv(output_path / f"{ticker.lower()}.csv", index=False)

    combined = combine_close_prices(asset_data)
    combined.to_csv(output_path / "combined_adj_close.csv")
    return output_path


def load_processed_data(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load saved ticker CSV files."""
    processed_path = Path(processed_dir)
    asset_data: dict[str, pd.DataFrame] = {}

    for csv_path in sorted(processed_path.glob("*.csv")):
        if csv_path.name == "combined_adj_close.csv":
            continue
        ticker = csv_path.stem.upper()
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        asset_data[ticker] = df

    return asset_data
