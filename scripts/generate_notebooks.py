"""Generate Week 9 challenge Jupyter notebooks."""

import json
from pathlib import Path


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


NOTEBOOKS = {
    "01_task1_eda.ipynb": [
        md(
            "# Task 1: Preprocess and Explore the Data\n\n"
            "**GMF Investments — Week 9 Challenge**\n\n"
            "Load TSLA, BND, and SPY data (Jan 2015 – Jun 2026), clean it, perform EDA, "
            "test stationarity, and compute risk metrics."
        ),
        code(
            "import sys\n"
            "from pathlib import Path\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n\n"
            "from src.data_loader import combine_close_prices, fetch_all_assets, save_processed_data\n"
            "from src.eda import (\n"
            "    plot_daily_returns,\n"
            "    plot_price_history,\n"
            "    plot_return_distribution,\n"
            "    plot_rolling_volatility,\n"
            "    run_adf_test,\n"
            "    summarize_adf_results,\n"
            ")\n"
            "from src.preprocessing import (\n"
            "    calculate_sharpe_ratio,\n"
            "    calculate_var,\n"
            "    clean_asset_frame,\n"
            "    compute_daily_returns,\n"
            "    detect_return_outliers,\n"
            ")\n\n"
            "sns.set_theme(style='whitegrid')\n"
            "DATA_DIR = PROJECT_ROOT / 'data' / 'processed'"
        ),
        md("## 1. Extract Historical Financial Data"),
        code(
            "asset_data = fetch_all_assets()\n"
            "asset_data = {t: clean_asset_frame(df) for t, df in asset_data.items()}\n"
            "save_processed_data(asset_data, DATA_DIR)\n\n"
            "prices = combine_close_prices(asset_data)\n"
            "prices.head()"
        ),
        md("## 2. Data Quality Summary"),
        code(
            "quality = []\n"
            "for ticker, df in asset_data.items():\n"
            "    quality.append({\n"
            "        'Ticker': ticker,\n"
            "        'Rows': len(df),\n"
            "        'Start': df['Date'].min().date(),\n"
            "        'End': df['Date'].max().date(),\n"
            "        'Missing Values': int(df.isna().sum().sum()),\n"
            "    })\n"
            "pd.DataFrame(quality)"
        ),
        md(
            "**Data quality actions taken:**\n"
            "- Ensured numeric dtypes for OHLCV columns\n"
            "- Sorted by date and forward-filled minor gaps\n"
            "- Combined adjusted close prices into a single wide frame for analysis"
        ),
        md("## 3. Exploratory Data Analysis"),
        code(
            "returns = prices.pct_change().dropna()\n\n"
            "fig1 = plot_price_history(prices, 'Adjusted Close Prices (2015–2026)')\n"
            "plt.show()\n\n"
            "fig2 = plot_daily_returns(returns, 'Daily Percentage Returns')\n"
            "plt.show()\n\n"
            "fig3 = plot_rolling_volatility(returns, window=30)\n"
            "plt.show()\n\n"
            "fig4 = plot_return_distribution(returns['TSLA'], 'TSLA')\n"
            "plt.show()"
        ),
        md("## 4. Outlier Detection"),
        code(
            "tsla_outliers = detect_return_outliers(returns['TSLA'])\n"
            "tsla_outliers.sort_values('ZScore', key=abs, ascending=False).head(10)"
        ),
        md("## 5. Stationarity Tests (ADF)"),
        code(
            "adf_results = [\n"
            "    run_adf_test(prices['TSLA'], 'TSLA Close'),\n"
            "    run_adf_test(returns['TSLA'], 'TSLA Daily Returns'),\n"
            "    run_adf_test(prices['SPY'], 'SPY Close'),\n"
            "    run_adf_test(returns['BND'], 'BND Daily Returns'),\n"
            "]\n"
            "summarize_adf_results(adf_results)"
        ),
        md(
            "**Interpretation:** Closing prices are typically non-stationary (unit root), "
            "while daily returns are usually stationary. Non-stationary price series require "
            "differencing (`d` in ARIMA) before modeling."
        ),
        md("## 6. Risk Metrics"),
        code(
            "risk_rows = []\n"
            "for ticker in prices.columns:\n"
            "    r = returns[ticker]\n"
            "    risk_rows.append({\n"
            "        'Asset': ticker,\n"
            "        'VaR (95%)': calculate_var(r),\n"
            "        'Sharpe Ratio': calculate_sharpe_ratio(r),\n"
            "        'Mean Daily Return': r.mean(),\n"
            "        'Daily Volatility': r.std(),\n"
            "    })\n"
            "pd.DataFrame(risk_rows)"
        ),
        md(
            "### Key Insights\n"
            "- **TSLA** shows strong upward drift with high volatility — high risk / high return profile.\n"
            "- **BND** exhibits lower daily fluctuations, supporting its role as a stabilizer.\n"
            "- **SPY** provides diversified market exposure with moderate volatility.\n"
            "- Returns are more suitable for risk measurement and ARIMA modeling than raw prices."
        ),
    ],
    "02_task2_forecasting.ipynb": [
        md("# Task 2: Build Time Series Forecasting Models\n\nCompare ARIMA/SARIMA and LSTM on Tesla (TSLA) price forecasting."),
        code(
            "import sys\nfrom pathlib import Path\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
            "from sklearn.preprocessing import MinMaxScaler\n"
            "from src.data_loader import combine_close_prices, fetch_all_assets\n"
            "from src.forecasting import (\n"
            "    build_lstm_model,\n"
            "    chronological_split,\n"
            "    compute_metrics,\n"
            "    create_lstm_sequences,\n"
            "    fit_auto_arima,\n"
            "    forecast_arima,\n"
            "    metrics_to_frame,\n"
            "    predict_lstm,\n"
            "    train_lstm,\n"
            ")\n"
            "from src.preprocessing import clean_asset_frame\n\n"
            "LOOKBACK = 60\nTRAIN_END = '2024-12-31'"
        ),
        code(
            "asset_data = {t: clean_asset_frame(df) for t, df in fetch_all_assets().items()}\n"
            "tsla = combine_close_prices(asset_data)['TSLA'].dropna()\n"
            "train, test = chronological_split(tsla, TRAIN_END)\n"
            "print(f'Train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)')\n"
            "print(f'Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)')"
        ),
        md("## ARIMA / SARIMA"),
        code(
            "arima_model = fit_auto_arima(train, seasonal=True, m=5)\n"
            "print(arima_model.summary())\n"
            "arima_pred = forecast_arima(arima_model, len(test))\n"
            "arima_metrics = compute_metrics(test.values, arima_pred)\n"
            "metrics_to_frame('ARIMA/SARIMA', arima_metrics)"
        ),
        md("## LSTM"),
        code(
            "scaler = MinMaxScaler()\n"
            "train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))\n"
            "full_scaled = scaler.transform(tsla.values.reshape(-1, 1))\n\n"
            "x_all, y_all = create_lstm_sequences(full_scaled, LOOKBACK)\n"
            "train_size = len(train) - LOOKBACK\n"
            "x_train, y_train = x_all[:train_size], y_all[:train_size]\n"
            "x_test = x_all[train_size:train_size + len(test)]\n\n"
            "lstm_model = build_lstm_model(LOOKBACK, units=50, learning_rate=0.001)\n"
            "train_lstm(lstm_model, x_train.reshape(-1, LOOKBACK, 1), y_train, epochs=25, batch_size=32)\n\n"
            "lstm_pred_scaled = predict_lstm(lstm_model, x_test.reshape(-1, LOOKBACK, 1))\n"
            "lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()\n"
            "lstm_metrics = compute_metrics(test.values[:len(lstm_pred)], lstm_pred)\n"
            "metrics_to_frame('LSTM', lstm_metrics)"
        ),
        md("## Model Comparison"),
        code(
            "comparison = pd.concat([\n"
            "    metrics_to_frame('ARIMA/SARIMA', arima_metrics),\n"
            "    metrics_to_frame('LSTM', lstm_metrics),\n"
            "], ignore_index=True)\n"
            "comparison"
        ),
        code(
            "plt.figure(figsize=(12, 5))\n"
            "plt.plot(test.index, test.values, label='Actual', linewidth=2)\n"
            "plt.plot(test.index, arima_pred, label='ARIMA/SARIMA', alpha=0.8)\n"
            "plt.plot(test.index[:len(lstm_pred)], lstm_pred, label='LSTM', alpha=0.8)\n"
            "plt.title('TSLA Test Set Forecast Comparison')\n"
            "plt.legend()\n"
            "plt.grid(alpha=0.3)\n"
            "plt.show()"
        ),
        md(
            "**Selection rationale:** Compare RMSE and MAPE. ARIMA is interpretable and fast; "
            "LSTM can capture nonlinear patterns but needs more data and tuning. "
            "The model with lower test RMSE is preferred for Task 3."
        ),
    ],
    "03_task3_future_forecasts.ipynb": [
        md("# Task 3: Forecast Future Market Trends\n\n6–12 month TSLA forecast with confidence intervals."),
        code(
            "import sys\nfrom pathlib import Path\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
            "from src.data_loader import combine_close_prices, fetch_all_assets\n"
            "from src.forecasting import fit_auto_arima, forecast_arima_with_intervals\n"
            "from src.preprocessing import clean_asset_frame\n\n"
            "FORECAST_DAYS = 252  # ~12 months of trading days"
        ),
        code(
            "asset_data = {t: clean_asset_frame(df) for t, df in fetch_all_assets().items()}\n"
            "tsla = combine_close_prices(asset_data)['TSLA'].dropna()\n"
            "train = tsla.loc[:'2024-12-31']\n\n"
            "arima_model = fit_auto_arima(train, seasonal=True, m=5)\n"
            "order = arima_model.order\n"
            "seasonal_order = arima_model.seasonal_order\n"
            "print('Selected order:', order, 'Seasonal:', seasonal_order)\n\n"
            "forecast_mean, forecast_lower, forecast_upper = forecast_arima_with_intervals(\n"
            "    train, steps=FORECAST_DAYS, order=order, seasonal_order=seasonal_order\n"
            ")\n"
            "future_index = pd.bdate_range(start=train.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)\n"
            "forecast_mean.index = future_index\n"
            "forecast_lower.index = future_index\n"
            "forecast_upper.index = future_index"
        ),
        code(
            "plt.figure(figsize=(14, 6))\n"
            "plt.plot(tsla.index, tsla.values, label='Historical', color='black')\n"
            "plt.plot(forecast_mean.index, forecast_mean.values, label='Forecast', color='royalblue')\n"
            "plt.fill_between(forecast_mean.index, forecast_lower, forecast_upper, alpha=0.2, color='royalblue', label='95% CI')\n"
            "plt.title('TSLA 12-Month Forecast with Confidence Intervals')\n"
            "plt.xlabel('Date')\n"
            "plt.ylabel('Price ($)')\n"
            "plt.legend()\n"
            "plt.grid(alpha=0.3)\n"
            "plt.show()"
        ),
        md(
            "## Trend Analysis\n\n"
            "The forecast path reflects recent momentum in Tesla's price series. Confidence intervals "
            "widen as the horizon increases, indicating rising uncertainty for longer horizons — "
            "consistent with EMH limitations on price prediction.\n\n"
            "**Opportunities:** Upward forecast slope may support a growth tilt for risk-tolerant clients.\n\n"
            "**Risks:** Wide intervals imply substantial downside scenarios; TSLA volatility remains elevated.\n\n"
            "**Reliability:** Near-term (1–3 month) forecasts are more reliable than 6–12 month projections."
        ),
    ],
    "04_task4_portfolio_optimization.ipynb": [
        md("# Task 4: Optimize Portfolio Based on Forecast\n\nMPT efficient frontier with TSLA forecast view."),
        code(
            "import sys\nfrom pathlib import Path\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\n\n"
            "from src.data_loader import combine_close_prices, fetch_all_assets\n"
            "from src.forecasting import fit_auto_arima, forecast_arima\n"
            "from src.portfolio import (\n"
            "    annualized_forecast_return_from_prices,\n"
            "    build_expected_returns_vector,\n"
            "    compute_covariance_matrix,\n"
            "    optimize_portfolios,\n"
            "    simulate_efficient_frontier,\n"
            ")\n"
            "from src.preprocessing import clean_asset_frame, compute_daily_returns\n"
        ),
        code(
            "asset_data = {t: clean_asset_frame(df) for t, df in fetch_all_assets().items()}\n"
            "prices = combine_close_prices(asset_data)\n"
            "returns = prices.pct_change().dropna()\n\n"
            "train_prices = prices.loc[:'2024-12-31']\n"
            "train_returns = returns.loc[:'2024-12-31']\n\n"
            "tsla_train = train_prices['TSLA']\n"
            "arima_model = fit_auto_arima(tsla_train, seasonal=True, m=5)\n"
            "future_prices = forecast_arima(arima_model, 252)\n"
            "tsla_expected = annualized_forecast_return_from_prices(\n"
            "    current_price=float(tsla_train.iloc[-1]),\n"
            "    forecast_prices=future_prices,\n"
            "    horizon_days=252,\n"
            ")\n\n"
            "expected_returns = build_expected_returns_vector(train_returns, tsla_expected)\n"
            "cov_matrix = compute_covariance_matrix(train_returns)\n"
            "expected_returns, cov_matrix"
        ),
        code(
            "plt.figure(figsize=(6, 5))\n"
            "sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='Blues')\n"
            "plt.title('Covariance Matrix (Daily Returns)')\n"
            "plt.show()"
        ),
        code(
            "results = optimize_portfolios(expected_returns, cov_matrix)\n"
            "frontier = simulate_efficient_frontier(expected_returns, cov_matrix)\n\n"
            "max_sharpe = results['max_sharpe']\n"
            "min_vol = results['min_volatility']\n"
            "max_sharpe['weights'], max_sharpe['performance']"
        ),
        code(
            "plt.figure(figsize=(10, 6))\n"
            "plt.plot(frontier['Volatility'], frontier['Expected Return'], label='Efficient Frontier')\n"
            "plt.scatter(max_sharpe['performance'][1], max_sharpe['performance'][0], color='green', s=120, label='Max Sharpe')\n"
            "plt.scatter(min_vol['performance'][1], min_vol['performance'][0], color='red', s=120, label='Min Volatility')\n"
            "plt.xlabel('Volatility (Risk)')\n"
            "plt.ylabel('Expected Return')\n"
            "plt.title('Efficient Frontier')\n"
            "plt.legend()\n"
            "plt.grid(alpha=0.3)\n"
            "plt.show()"
        ),
        md(
            "## Recommendation\n\n"
            "We recommend the **Maximum Sharpe Ratio** portfolio for clients seeking strong "
            "risk-adjusted returns while maintaining diversification through BND and SPY. "
            "See optimized weights and performance metrics above."
        ),
    ],
    "05_task5_backtesting.ipynb": [
        md("# Task 5: Strategy Backtesting\n\nCompare optimized portfolio vs 60% SPY / 40% BND benchmark."),
        code(
            "import sys\nfrom pathlib import Path\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
            "import matplotlib.pyplot as plt\nimport pandas as pd\n\n"
            "from src.backtesting import compare_strategies\n"
            "from src.data_loader import combine_close_prices, fetch_all_assets\n"
            "from src.portfolio import build_expected_returns_vector, compute_covariance_matrix, optimize_portfolios\n"
            "from src.forecasting import fit_auto_arima, forecast_arima\n"
            "from src.preprocessing import clean_asset_frame\n"
            "from src.portfolio import annualized_forecast_return_from_prices\n"
        ),
        code(
            "asset_data = {t: clean_asset_frame(df) for t, df in fetch_all_assets().items()}\n"
            "prices = combine_close_prices(asset_data)\n"
            "returns = prices.pct_change().dropna()\n\n"
            "train_returns = returns.loc[:'2024-12-31']\n"
            "backtest_returns = returns.loc['2025-01-01':'2026-01-31']\n\n"
            "tsla_train = prices.loc[:'2024-12-31', 'TSLA']\n"
            "arima_model = fit_auto_arima(tsla_train, seasonal=True, m=5)\n"
            "future_prices = forecast_arima(arima_model, 252)\n"
            "tsla_expected = annualized_forecast_return_from_prices(\n"
            "    float(tsla_train.iloc[-1]), future_prices, 252\n"
            ")\n\n"
            "expected_returns = build_expected_returns_vector(train_returns, tsla_expected)\n"
            "cov_matrix = compute_covariance_matrix(train_returns)\n"
            "strategy_weights = optimize_portfolios(expected_returns, cov_matrix)['max_sharpe']['weights']\n"
            "benchmark_weights = {'TSLA': 0.0, 'SPY': 0.6, 'BND': 0.4}\n\n"
            "metrics, strat_cum, bench_cum = compare_strategies(\n"
            "    backtest_returns, strategy_weights, benchmark_weights\n"
            ")\n"
            "metrics"
        ),
        code(
            "plt.figure(figsize=(12, 5))\n"
            "plt.plot(strat_cum.index, strat_cum.values, label='Optimized Strategy')\n"
            "plt.plot(bench_cum.index, bench_cum.values, label='60/40 SPY-BND Benchmark')\n"
            "plt.title('Cumulative Returns: Strategy vs Benchmark')\n"
            "plt.ylabel('Growth of $1')\n"
            "plt.legend()\n"
            "plt.grid(alpha=0.3)\n"
            "plt.show()"
        ),
        md(
            "## Conclusion\n\n"
            "This backtest compares a forecast-informed optimal portfolio against a passive 60/40 benchmark "
            "over Jan 2025 – Jan 2026. Outperformance would suggest the model-driven allocation adds value; "
            "underperformance highlights EMH constraints and the limits of a single-asset forecast view.\n\n"
            "**Limitations:** Static weights, no transaction costs, short window, and look-ahead bias in "
            "covariance estimation if not carefully separated."
        ),
    ],
}


def main():
    out_dir = Path(__file__).resolve().parents[1] / "notebooks"
    for name, cells in NOTEBOOKS.items():
        path = out_dir / name
        path.write_text(json.dumps(nb(cells), indent=1))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
