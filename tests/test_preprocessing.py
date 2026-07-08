"""Unit tests for portfolio optimization project."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import build_weighted_returns, cumulative_returns, maximum_drawdown
from src.forecasting import compute_metrics, create_lstm_sequences
from src.preprocessing import (
    calculate_sharpe_ratio,
    calculate_var,
    compute_daily_returns,
    compute_rolling_volatility,
)


@pytest.fixture
def sample_returns():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    values = np.random.default_rng(42).normal(0.001, 0.02, size=100)
    return pd.Series(values, index=dates, name="TSLA")


def test_compute_daily_returns(sample_returns):
    prices = (1 + sample_returns).cumprod() * 100
    returns = compute_daily_returns(prices)
    assert len(returns) == len(prices) - 1
    assert returns.notna().all()


def test_rolling_volatility(sample_returns):
    vol = compute_rolling_volatility(sample_returns, window=10)
    assert vol.dropna().shape[0] > 0


def test_calculate_var(sample_returns):
    var = calculate_var(sample_returns, confidence=0.95)
    assert var < 0


def test_sharpe_ratio(sample_returns):
    sharpe = calculate_sharpe_ratio(sample_returns)
    assert isinstance(sharpe, float)


def test_compute_metrics():
    y_true = np.array([100, 102, 101, 105])
    y_pred = np.array([100, 103, 100, 104])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.mae >= 0
    assert metrics.rmse >= 0
    assert metrics.mape >= 0


def test_create_lstm_sequences():
    values = np.arange(100).reshape(-1, 1).astype(float)
    x, y = create_lstm_sequences(values, lookback=10)
    assert x.shape == (90, 10)
    assert y.shape == (90,)


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


def test_cumulative_returns(sample_returns):
    cumulative = cumulative_returns(sample_returns)
    assert cumulative.iloc[-1] > 0
