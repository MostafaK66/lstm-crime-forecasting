"""Service orchestration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crime_forecasting.backend import ModelBackend, ModelResult
from crime_forecasting.config import AppConfig
from crime_forecasting.service import ForecastService, model_path
from crime_forecasting.windows import prepare_windows


def test_orchestrates_injected_boundaries(
    app_config: AppConfig, daily_frame: pd.DataFrame
) -> None:
    prepared = prepare_windows(daily_frame, app_config.window)
    predictions = np.zeros((prepared.validation.sample_count, 2))
    backend = ModelBackend(
        lambda *_: ModelResult(predictions, {}, None, Path("model.keras"))
    )
    written: list[object] = []

    def writer(*args: object, **kwargs: object) -> tuple[Path, ...]:
        written.extend((args, kwargs))
        return (Path("result.csv"),)

    service = ForecastService(
        loader=lambda _: daily_frame,
        preparer=lambda *_: prepared,
        backend=backend,
        writer=writer,
    )
    summary = service.run(app_config)
    assert summary.training_samples == prepared.train.sample_count
    assert summary.validation_samples == prepared.validation.sample_count
    assert summary.artifact_paths == (Path("result.csv"),)
    assert written
    assert model_path(app_config.output).name == "model.keras"
