"""Modern Portfolio Theory optimization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


def compute_expected_returns_from_history(
    daily_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """Annualized expected returns from historical mean daily returns."""
    return daily_returns.mean() * periods_per_year


def compute_covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix of daily returns."""
    return daily_returns.cov()


def build_expected_returns_vector(
    historical_returns: pd.DataFrame,
    tsla_forecast_return: float,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Build expected returns vector mixing forecast (TSLA) and history (BND, SPY).
    tsla_forecast_return should be annualized.
    """
    expected = compute_expected_returns_from_history(historical_returns, periods_per_year)
    expected["TSLA"] = tsla_forecast_return
    return expected


def optimize_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> dict:
    """Find max Sharpe and min volatility portfolios."""
    ef_max_sharpe = EfficientFrontier(expected_returns, cov_matrix)
    max_sharpe_weights = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    max_sharpe_perf = ef_max_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)

    ef_min_vol = EfficientFrontier(expected_returns, cov_matrix)
    min_vol_weights = ef_min_vol.min_volatility()
    min_vol_perf = ef_min_vol.portfolio_performance(risk_free_rate=risk_free_rate)

    return {
        "max_sharpe": {
            "weights": ef_max_sharpe.clean_weights(),
            "performance": max_sharpe_perf,
        },
        "min_volatility": {
            "weights": ef_min_vol.clean_weights(),
            "performance": min_vol_perf,
        },
    }


def simulate_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    points: int = 50,
) -> pd.DataFrame:
    """Generate risk/return points along the efficient frontier."""
    targets = np.linspace(expected_returns.min(), expected_returns.max() * 1.05, points)
    rows = []

    for target in targets:
        try:
            ef = EfficientFrontier(expected_returns, cov_matrix)
            ef.efficient_return(float(target))
            ann_ret, ann_vol, sharpe = ef.portfolio_performance()
            rows.append(
                {
                    "Expected Return": ann_ret,
                    "Volatility": ann_vol,
                    "Sharpe Ratio": sharpe,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def annualized_forecast_return_from_prices(
    current_price: float,
    forecast_prices: np.ndarray,
    horizon_days: int,
    periods_per_year: int = 252,
) -> float:
    """Estimate annualized expected return from price forecast."""
    terminal_price = float(forecast_prices[-1])
    total_return = (terminal_price / current_price) - 1
    years = horizon_days / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def discrete_allocation_from_weights(
    weights: dict,
    latest_prices: pd.Series,
    total_portfolio_value: float,
) -> dict:
    """Convert fractional weights to share counts."""
    allocator = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = allocator.greedy_portfolio()
    return {"allocation": allocation, "leftover_cash": leftover}
