"""Exploratory data analysis utilities."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


def run_adf_test(series: pd.Series, name: str = "series") -> dict:
    """Run Augmented Dickey-Fuller test and return structured results."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lags": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def summarize_adf_results(results: Iterable[dict]) -> pd.DataFrame:
    """Convert ADF test outputs into a comparison table."""
    rows = []
    for result in results:
        rows.append(
            {
                "Series": result["name"],
                "ADF Statistic": result["adf_statistic"],
                "p-value": result["p_value"],
                "Stationary (p < 0.05)": result["is_stationary"],
            }
        )
    return pd.DataFrame(rows)


def plot_price_history(prices: pd.DataFrame, title: str = "Adjusted Close Prices") -> plt.Figure:
    """Plot closing prices for all assets."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in prices.columns:
        ax.plot(prices.index, prices[ticker], label=ticker)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_daily_returns(returns: pd.DataFrame, title: str = "Daily Returns") -> plt.Figure:
    """Plot daily percentage returns."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in returns.columns:
        ax.plot(returns.index, returns[ticker], label=ticker, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 30,
    title: str | None = None,
) -> plt.Figure:
    """Plot rolling volatility for each asset."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in returns.columns:
        rolling_std = returns[ticker].rolling(window).std()
        ax.plot(rolling_std.index, rolling_std, label=ticker)
    ax.set_title(title or f"{window}-Day Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std Dev")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_return_distribution(returns: pd.Series, ticker: str = "TSLA") -> plt.Figure:
    """Histogram and KDE of daily returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(returns.dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"{ticker} Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig
