"""Unit tests for forecasting utilities."""

import numpy as np

from src.forecasting import compute_metrics, create_lstm_sequences


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
