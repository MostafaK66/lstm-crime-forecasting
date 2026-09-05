"""Shared deterministic test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crime_forecasting.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    WindowConfig,
)


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, freq="D", name="date")
    return pd.DataFrame(
        {
            "count": np.arange(12, dtype=np.float64),
            "month": index.month,
            "weekday": index.dayofweek,
            "day": index.day,
        },
        index=index,
    )


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(tmp_path / "crime.csv", "YEAR", "MONTH", "DAY"),
        window=WindowConfig(4, 2, 1, 0.3),
        model=ModelConfig("grid", True, 33, 2, 2, 1, 0.0),
        output=OutputConfig(
            tmp_path / "output",
            tmp_path / "models",
            "model.keras",
            "predictions.csv",
            "metrics.csv",
            "residuals.csv",
            "run.json",
        ),
    )
