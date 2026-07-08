"""Unit tests for backtesting utilities."""

import numpy as np
import pandas as pd

from src.backtesting import build_weighted_returns, cumulative_returns, maximum_drawdown


def test_weighted_returns():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    returns = pd.DataFrame(
        {
            "TSLA": [0.01, -0.02, 0.03, 0.0, 0.01],
            "BND": [0.001, 0.002, -0.001, 0.0, 0.001],
            "SPY": [0.005, -0.01, 0.02, 0.001, 0.002],
        },
        index=dates,
    )
    weights = {"TSLA": 0.2, "BND": 0.3, "SPY": 0.5}
    portfolio = build_weighted_returns(returns, weights)
    assert len(portfolio) == 5
    assert np.isclose(weights["TSLA"] + weights["BND"] + weights["SPY"], 1.0)


def test_maximum_drawdown():
    cumulative = pd.Series([1.0, 1.1, 1.05, 1.2, 0.9])
    mdd = maximum_drawdown(cumulative)
    assert mdd < 0


def test_cumulative_returns():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.02, size=10), index=dates)
    cumulative = cumulative_returns(returns)
    assert cumulative.iloc[-1] > 0
