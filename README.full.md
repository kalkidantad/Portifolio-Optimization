# Portfolio Optimization — Week 9 Challenge

Time series forecasting and Modern Portfolio Theory for **GMF Investments**.

## Team
Kerod · Mahbubah · Feven

## Overview
This project forecasts Tesla (TSLA) prices using ARIMA/SARIMA and LSTM models, then uses those insights with bond (BND) and market (SPY) data to construct an optimized portfolio, visualize the efficient frontier, and backtest against a 60/40 benchmark.

## Project Structure
```
portfolio-optimization/
├── data/processed/          # Saved CSV data
├── notebooks/               # Task notebooks (01–05)
├── scripts/                 # Data download & utilities
├── src/                     # Reusable Python modules
└── tests/                   # Unit tests
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start
```bash
# Download and cache YFinance data
python scripts/download_data.py

# Generate notebooks (if needed)
python scripts/generate_notebooks.py

# Run tests
pytest tests/ -v
```

## Notebooks
| Notebook | Description |
|----------|-------------|
| `01_task1_eda.ipynb` | Data extraction, EDA, stationarity, VaR, Sharpe |
| `02_task2_forecasting.ipynb` | ARIMA/SARIMA vs LSTM comparison |
| `03_task3_future_forecasts.ipynb` | 12-month forecast with confidence intervals |
| `04_task4_portfolio_optimization.ipynb` | Efficient frontier & recommendation |
| `05_task5_backtesting.ipynb` | Strategy vs benchmark backtest |

## Assets
| Ticker | Role |
|--------|------|
| TSLA | High-growth, forecasted expected return |
| BND | Low-risk stability |
| SPY | Diversified market exposure |

## Data Period
January 1, 2015 — June 30, 2026 (via YFinance)

## Key Methods
- **Forecasting:** pmdarima `auto_arima`, TensorFlow LSTM
- **Portfolio:** PyPortfolioOpt efficient frontier
- **Evaluation:** MAE, RMSE, MAPE, Sharpe ratio, max drawdown

## License
Educational project — 10 Academy AI Mastery, Week 9.
