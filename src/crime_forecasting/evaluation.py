"""Forecast metrics and tidy output tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from crime_forecasting.errors import DataValidationError

FloatArray = npt.NDArray[np.float64]
DateArray = npt.NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """All deterministic evaluation artifacts."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    residuals: pd.DataFrame


def evaluate(
    actual: npt.ArrayLike,
    predicted: npt.ArrayLike,
    dates: DateArray,
    last_observed: npt.ArrayLike,
) -> Evaluation:
    """Compare a model with a persistence baseline for every forecast horizon."""
    actual_array = _matrix(actual, "actual")
    predicted_array = _matrix(predicted, "predicted")
    baseline_values = np.asarray(last_observed, dtype=np.float64)
    if actual_array.shape != predicted_array.shape:
        raise DataValidationError("Actual and predicted values must have the same shape")
    if dates.shape != actual_array.shape:
        raise DataValidationError("Forecast dates must match the value matrix")
    if baseline_values.shape != (actual_array.shape[0],):
        message = "Last-observed values must contain one value per sample"
        raise DataValidationError(message)
    if not np.isfinite(baseline_values).all():
        raise DataValidationError("Last-observed values contain a non-finite value")

    baseline = np.repeat(baseline_values[:, None], actual_array.shape[1], axis=1)
    horizon = np.arange(1, actual_array.shape[1] + 1, dtype=np.int64)
    model_mse = np.mean(np.square(actual_array - predicted_array), axis=0)
    baseline_mse = np.mean(np.square(actual_array - baseline), axis=0)
    metrics = pd.DataFrame(
        {"horizon": horizon, "model_mse": model_mse, "persistence_mse": baseline_mse}
    )

    sample = np.repeat(np.arange(actual_array.shape[0]), actual_array.shape[1])
    predictions = pd.DataFrame(
        {
            "sample": sample,
            "horizon": np.tile(horizon, actual_array.shape[0]),
            "date": pd.to_datetime(dates.reshape(-1)),
            "actual": actual_array.reshape(-1),
            "predicted": predicted_array.reshape(-1),
            "persistence": baseline.reshape(-1),
        }
    )
    residuals = pd.DataFrame(
        {
            "date": pd.to_datetime(dates[:, 0]),
            "residual": actual_array[:, 0] - predicted_array[:, 0],
        }
    )
    return Evaluation(predictions=predictions, metrics=metrics, residuals=residuals)


def _matrix(values: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2 or array.size == 0:
        raise DataValidationError(f"'{name}' must be a non-empty two-dimensional matrix")
    if not np.isfinite(array).all():
        raise DataValidationError(f"'{name}' contains a non-finite value")
    return array
