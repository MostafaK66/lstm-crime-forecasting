"""Safe supervised-window construction and chronological preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from crime_forecasting.config import WindowConfig
from crime_forecasting.errors import DataValidationError, InsufficientDataError
from crime_forecasting.scaling import StandardScaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
DateArray = npt.NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class WindowSet:
    """One model-ready partition of chronological samples."""

    inputs: dict[str, FloatArray | IntArray]
    targets: FloatArray
    dates: DateArray
    last_observed: FloatArray

    @property
    def sample_count(self) -> int:
        return self.targets.shape[0]


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Chronological train/validation data and target scaling metadata."""

    train: WindowSet
    validation: WindowSet
    target_scaler: StandardScaler


@dataclass(frozen=True, slots=True)
class _RawWindows:
    numeric: FloatArray
    past_calendar: IntArray
    future_calendar: IntArray
    targets: FloatArray
    dates: DateArray
    last_observed: FloatArray


def prepare_windows(frame: pd.DataFrame, config: WindowConfig) -> PreparedData:
    """Build explicit windows, split chronologically, and fit training-only scalers."""
    raw = _build_raw_windows(frame, config)
    split = int(raw.targets.shape[0] * (1.0 - config.validation_fraction))
    if split < 1 or split >= raw.targets.shape[0]:
        raise InsufficientDataError(
            "The configured validation fraction must leave at least one sample "
            "in both train and validation partitions"
        )

    numeric_scaler = StandardScaler.fit(raw.numeric[:split])
    target_scaler = StandardScaler.fit(raw.targets[:split])
    train = _partition(raw, slice(0, split), numeric_scaler, target_scaler)
    validation = _partition(raw, slice(split, None), numeric_scaler, target_scaler)
    return PreparedData(train=train, validation=validation, target_scaler=target_scaler)


def _build_raw_windows(frame: pd.DataFrame, config: WindowConfig) -> _RawWindows:
    required = {"count", "month", "weekday", "day"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Daily data is missing columns: {', '.join(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError("Daily data must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise DataValidationError("Daily dates must be unique and chronological")
    if frame.isna().any().any():
        raise DataValidationError("Daily data contains missing values")

    counts = frame["count"].to_numpy(dtype=np.float64)
    calendar = frame[["month", "weekday", "day"]].to_numpy(dtype=np.int64)
    dates = frame.index.to_numpy(dtype="datetime64[ns]")
    sample_span = config.history + config.horizon
    starts = range(0, len(frame) - sample_span + 1, config.step)
    numeric_windows: list[FloatArray] = []
    past_windows: list[IntArray] = []
    future_windows: list[IntArray] = []
    target_windows: list[FloatArray] = []
    date_windows: list[DateArray] = []
    last_observed: list[float] = []
    for start in starts:
        boundary = start + config.history
        end = boundary + config.horizon
        numeric_windows.append(counts[start:boundary, None])
        past_windows.append(calendar[start:boundary])
        future_windows.append(calendar[boundary:end])
        target_windows.append(counts[boundary:end, None])
        date_windows.append(dates[boundary:end])
        last_observed.append(float(counts[boundary - 1]))
    if not numeric_windows:
        raise InsufficientDataError(
            f"Need at least {sample_span} daily observations for "
            f"history={config.history} "
            f"and horizon={config.horizon}; received {len(frame)}"
        )
    return _RawWindows(
        numeric=np.stack(numeric_windows),
        past_calendar=np.stack(past_windows),
        future_calendar=np.stack(future_windows),
        targets=np.stack(target_windows),
        dates=np.stack(date_windows),
        last_observed=np.asarray(last_observed, dtype=np.float64),
    )


def _partition(
    raw: _RawWindows,
    selected: slice,
    numeric_scaler: StandardScaler,
    target_scaler: StandardScaler,
) -> WindowSet:
    past = raw.past_calendar[selected]
    future = raw.future_calendar[selected]
    inputs: dict[str, FloatArray | IntArray] = {
        "numeric_history": numeric_scaler.transform(raw.numeric[selected]),
        "past_month": past[:, :, 0],
        "past_weekday": past[:, :, 1],
        "past_day": past[:, :, 2],
        "future_month": future[:, :, 0],
        "future_weekday": future[:, :, 1],
        "future_day": future[:, :, 2],
    }
    return WindowSet(
        inputs=inputs,
        targets=target_scaler.transform(raw.targets[selected]),
        dates=raw.dates[selected],
        last_observed=raw.last_observed[selected],
    )
