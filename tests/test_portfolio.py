"""Unit tests for portfolio optimization utilities."""

import numpy as np
import pandas as pd

from src.portfolio import (
    annualized_forecast_return_from_prices,
    build_expected_returns_vector,
    compute_covariance_matrix,
    compute_expected_returns_from_history,
)


def _sample_asset_returns():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "TSLA": rng.normal(0.002, 0.03, size=60),
            "BND": rng.normal(0.0002, 0.005, size=60),
            "SPY": rng.normal(0.0008, 0.012, size=60),
        },
        index=dates,
    )


def test_compute_expected_returns_from_history():
    returns = _sample_asset_returns()
    expected = compute_expected_returns_from_history(returns)
    assert len(expected) == 3
    assert expected["TSLA"] > expected["BND"]


def test_compute_covariance_matrix():
    returns = _sample_asset_returns()
    cov = compute_covariance_matrix(returns)
    assert cov.shape == (3, 3)
    assert cov.loc["TSLA", "TSLA"] > 0


def test_build_expected_returns_vector():
    returns = _sample_asset_returns()
    expected = build_expected_returns_vector(returns, tsla_forecast_return=0.25)
    assert expected["TSLA"] == 0.25


def test_annualized_forecast_return_from_prices():
    forecast = np.linspace(100, 120, 252)
    annual_return = annualized_forecast_return_from_prices(100, forecast, 252)
    assert annual_return > 0
