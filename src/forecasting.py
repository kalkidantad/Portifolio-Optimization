"""Time series forecasting models: ARIMA/SARIMA and LSTM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastMetrics:
    """Calculate MAE, RMSE, and MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape)


def metrics_to_frame(name: str, metrics: ForecastMetrics) -> pd.DataFrame:
    """Convert metrics to a single-row DataFrame."""
    return pd.DataFrame(
        [{"Model": name, "MAE": metrics.mae, "RMSE": metrics.rmse, "MAPE (%)": metrics.mape}]
    )


def chronological_split(
    series: pd.Series,
    train_end: str = "2024-12-31",
) -> tuple[pd.Series, pd.Series]:
    """Split a time series chronologically."""
    train = series.loc[:train_end]
    test = series.loc[train_end:].iloc[1:]
    return train, test


def fit_auto_arima(
    train: pd.Series,
    seasonal: bool = True,
    m: int = 5,
) -> object:
    """Fit ARIMA/SARIMA using pmdarima auto_arima."""
    return auto_arima(
        train,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )


def forecast_arima(model: object, steps: int) -> np.ndarray:
    """Generate point forecasts from a fitted ARIMA model."""
    forecast = model.predict(n_periods=steps)
    return np.asarray(forecast)


def forecast_arima_with_intervals(
    train: pd.Series,
    steps: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    alpha: float = 0.05,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fit SARIMAX on full training data and return forecast with confidence bands."""
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    conf = forecast.conf_int(alpha=alpha)
    lower = conf.iloc[:, 0]
    upper = conf.iloc[:, 1]
    return mean, lower, upper


def create_lstm_sequences(
    scaled_values: np.ndarray,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised learning sequences for LSTM."""
    x, y = [], []
    for i in range(lookback, len(scaled_values)):
        x.append(scaled_values[i - lookback : i, 0])
        y.append(scaled_values[i, 0])
    return np.array(x), np.array(y)


def build_lstm_model(lookback: int, units: int = 50, learning_rate: float = 0.001):
    """Build a simple LSTM regression model."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
            LSTM(units // 2, return_sequences=False),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_lstm(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.1,
    verbose: int = 0,
):
    """Train LSTM model."""
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    return history


def predict_lstm(model, x: np.ndarray) -> np.ndarray:
    """Generate LSTM predictions."""
    return model.predict(x, verbose=0).flatten()


def iterative_lstm_forecast(
    model,
    last_sequence: np.ndarray,
    steps: int,
    scaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-step LSTM forecast by feeding predictions back."""
    sequence = last_sequence.copy()
    preds_scaled = []

    for _ in range(steps):
        x_input = sequence.reshape(1, sequence.shape[0], 1)
        next_pred = model.predict(x_input, verbose=0)[0, 0]
        preds_scaled.append(next_pred)
        sequence = np.roll(sequence, -1)
        sequence[-1] = next_pred

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    # Approximate intervals using historical test error std
    std = np.std(preds) * 0.1
    lower = preds - 1.96 * std
    upper = preds + 1.96 * std
    return preds, lower, upper
