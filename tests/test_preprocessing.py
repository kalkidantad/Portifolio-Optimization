"""Unit tests for data preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

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
