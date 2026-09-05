"""Crime-record ingestion and daily aggregation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from crime_forecasting.config import DataConfig
from crime_forecasting.errors import DataValidationError

CsvReader = Callable[..., pd.DataFrame]


def load_daily_counts(
    config: DataConfig, *, reader: CsvReader = pd.read_csv
) -> pd.DataFrame:
    """Read records, validate date fields, and fill missing calendar days with zero."""
    try:
        records = reader(config.path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        message = f"Cannot read crime data '{config.path}': {error}"
        raise DataValidationError(message) from error

    required = [config.year_column, config.month_column, config.day_column]
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise DataValidationError(f"Missing required date columns: {', '.join(missing)}")
    if records.empty:
        raise DataValidationError("Crime data contains no records")

    date_parts: dict[str, pd.Series[float]] = {}
    for standard, column in zip(("year", "month", "day"), required, strict=True):
        numeric = pd.to_numeric(records[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            message = f"Column '{column}' must contain whole date numbers"
            raise DataValidationError(message)
        date_parts[standard] = numeric
    try:
        date_frame = pd.DataFrame(date_parts)
        dates = pd.to_datetime(date_frame, errors="raise")
    except (ValueError, OverflowError) as error:
        message = f"Crime data contains an invalid calendar date: {error}"
        raise DataValidationError(message) from error

    counts = pd.Series(1, index=pd.DatetimeIndex(dates), dtype="int64")
    daily = counts.groupby(level=0).sum().sort_index()
    complete_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(complete_index, fill_value=0)
    index = pd.DatetimeIndex(complete_index)
    frame = pd.DataFrame({"count": daily.astype("float64")}, index=index)
    frame.index.name = "date"
    frame["month"] = index.month.astype("int64")
    frame["weekday"] = index.dayofweek.astype("int64")
    frame["day"] = index.day.astype("int64")
    return frame
