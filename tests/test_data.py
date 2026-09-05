"""Data ingestion tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crime_forecasting.config import DataConfig
from crime_forecasting.data import load_daily_counts
from crime_forecasting.errors import DataValidationError


def config() -> DataConfig:
    return DataConfig(Path("unused.csv"), "Y", "M", "D")


def test_aggregates_and_inserts_zero_days() -> None:
    records = pd.DataFrame({"Y": [2024, 2024, 2024], "M": [1, 1, 1], "D": [1, 1, 3]})
    result = load_daily_counts(config(), reader=lambda _: records)
    assert result["count"].tolist() == [2.0, 0.0, 1.0]
    assert result["weekday"].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (pd.DataFrame(), "Missing required"),
        (pd.DataFrame(columns=["Y", "M", "D"]), "no records"),
        (pd.DataFrame({"Y": [2024], "M": [1], "D": [1.5]}), "whole date"),
        (pd.DataFrame({"Y": [2024], "M": [13], "D": [1]}), "invalid calendar"),
    ],
)
def test_rejects_invalid_data(records: pd.DataFrame, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        load_daily_counts(config(), reader=lambda _: records)


def test_wraps_reader_error() -> None:
    def fail(_: Path) -> pd.DataFrame:
        raise OSError("unavailable")

    with pytest.raises(DataValidationError, match="Cannot read crime data"):
        load_daily_counts(config(), reader=fail)
