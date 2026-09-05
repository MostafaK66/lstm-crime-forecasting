"""Window construction tests."""

from __future__ import annotations

import pandas as pd
import pytest

from crime_forecasting.config import WindowConfig
from crime_forecasting.errors import DataValidationError, InsufficientDataError
from crime_forecasting.windows import prepare_windows


def test_builds_chronological_windows_without_leakage(daily_frame: pd.DataFrame) -> None:
    result = prepare_windows(daily_frame, WindowConfig(4, 2, 2, 0.34))
    assert result.train.sample_count == 2
    assert result.validation.sample_count == 2
    assert result.validation.last_observed.tolist() == [7.0, 9.0]
    assert result.validation.dates[0, 0] == daily_frame.index[8].to_datetime64()
    assert set(result.train.inputs) == {
        "numeric_history",
        "past_month",
        "past_weekday",
        "past_day",
        "future_month",
        "future_weekday",
        "future_day",
    }


def test_requires_enough_daily_observations(daily_frame: pd.DataFrame) -> None:
    with pytest.raises(InsufficientDataError, match="Need at least"):
        prepare_windows(daily_frame.iloc[:4], WindowConfig(4, 2, 1, 0.3))


def test_requires_nonempty_partitions(daily_frame: pd.DataFrame) -> None:
    with pytest.raises(InsufficientDataError, match="both train and validation"):
        prepare_windows(daily_frame.iloc[:6], WindowConfig(4, 2, 1, 0.3))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns="day"),
        lambda frame: frame.reset_index(drop=True),
        lambda frame: frame.sort_index(ascending=False),
        lambda frame: frame.assign(count=float("nan")),
    ],
)
def test_rejects_invalid_daily_frame(daily_frame: pd.DataFrame, mutator: object) -> None:
    changed = mutator(daily_frame)  # type: ignore[operator]
    with pytest.raises(DataValidationError):
        prepare_windows(changed, WindowConfig(4, 2, 1, 0.3))
