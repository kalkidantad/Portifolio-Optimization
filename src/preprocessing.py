"""Data cleaning, feature engineering, and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def clean_asset_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dtypes, sort by date, and forward-fill missing values."""
    cleaned = df.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned[numeric_cols] = cleaned[numeric_cols].ffill().bfill()
    return cleaned


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily percentage returns."""
    return prices.pct_change().dropna()


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Rolling standard deviation of daily returns."""
    return returns.rolling(window=window).std()


def detect_return_outliers(
    returns: pd.Series,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag unusually high or low daily returns using z-scores."""
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = returns.loc[z_scores.abs() >= z_threshold]
    return pd.DataFrame(
        {
            "Date": outliers.index,
            "Return": outliers.values,
            "ZScore": z_scores.loc[outliers.index].values,
        }
    )


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk at the given confidence level."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio from daily returns."""
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def annualize_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Convert mean daily return to annualized return."""
    return float(daily_returns.mean() * periods_per_year)


def annualize_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Convert daily return volatility to annualized volatility."""
    return float(daily_returns.std() * np.sqrt(periods_per_year))


def scale_series(
    series: pd.Series,
    method: str = "minmax",
) -> tuple[np.ndarray, object]:
    """Scale a series for ML models; returns scaled values and fitted scaler."""
    values = series.values.reshape(-1, 1)
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Create returns and rolling volatility features for all assets."""
    features = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        returns = compute_daily_returns(prices[ticker])
        features[f"{ticker}_return"] = returns
        features[f"{ticker}_volatility_30d"] = compute_rolling_volatility(returns, 30)
    return features.dropna()
