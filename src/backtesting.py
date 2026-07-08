"""Portfolio backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import annualize_return, annualize_volatility, calculate_sharpe_ratio


def build_weighted_returns(
    daily_returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute portfolio daily returns from asset returns and weights."""
    weight_series = pd.Series(weights).reindex(daily_returns.columns).fillna(0.0)
    return daily_returns.mul(weight_series, axis=1).sum(axis=1)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert daily returns to cumulative growth of $1."""
    return (1 + returns).cumprod()


def maximum_drawdown(cumulative: pd.Series) -> float:
    """Maximum peak-to-trough decline."""
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    return float(drawdown.min())


def summarize_backtest(
    returns: pd.Series,
    name: str,
    risk_free_rate: float = 0.02,
) -> dict:
    """Compute standard backtest performance metrics."""
    cumulative = cumulative_returns(returns)
    total_return = float(cumulative.iloc[-1] - 1)
    return {
        "Portfolio": name,
        "Total Return": total_return,
        "Annualized Return": annualize_return(returns),
        "Annualized Volatility": annualize_volatility(returns),
        "Sharpe Ratio": calculate_sharpe_ratio(returns, risk_free_rate / 252),
        "Maximum Drawdown": maximum_drawdown(cumulative),
    }


def compare_strategies(
    daily_returns: pd.DataFrame,
    strategy_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    strategy_name: str = "Optimized Strategy",
    benchmark_name: str = "60/40 Benchmark",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Compare strategy vs benchmark over a return window."""
    strategy_returns = build_weighted_returns(daily_returns, strategy_weights)
    benchmark_returns = build_weighted_returns(daily_returns, benchmark_weights)

    metrics = pd.DataFrame(
        [
            summarize_backtest(strategy_returns, strategy_name),
            summarize_backtest(benchmark_returns, benchmark_name),
        ]
    ).set_index("Portfolio")

    return metrics, cumulative_returns(strategy_returns), cumulative_returns(benchmark_returns)


def monthly_rebalance_returns(
    daily_returns: pd.DataFrame,
    target_weights: dict[str, float],
) -> pd.Series:
    """Simulate monthly rebalancing back to target weights."""
    weight_series = pd.Series(target_weights).reindex(daily_returns.columns).fillna(0.0)
    portfolio_returns = []

    for _, month_returns in daily_returns.groupby(pd.Grouper(freq="ME")):
        if month_returns.empty:
            continue
        month_portfolio = month_returns.mul(weight_series, axis=1).sum(axis=1)
        portfolio_returns.append(month_portfolio)

    if not portfolio_returns:
        return pd.Series(dtype=float)

    return pd.concat(portfolio_returns).sort_index()
