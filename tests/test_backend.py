"""Model boundary tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crime_forecasting.backend import ModelBackend, ModelResult
from crime_forecasting.config import ModelConfig, OutputConfig, WindowConfig
from crime_forecasting.errors import ModelBackendError
from crime_forecasting.windows import prepare_windows


def model_config() -> ModelConfig:
    return ModelConfig("grid", True, 1, 1, 1, 0, 0.0)


def output_config(tmp_path: Path) -> OutputConfig:
    return OutputConfig(
        tmp_path, tmp_path, "model.keras", "p.csv", "m.csv", "r.csv", "run.json"
    )


def test_accepts_valid_injected_result(daily_frame: pd.DataFrame, tmp_path: Path) -> None:
    data = prepare_windows(daily_frame, WindowConfig(4, 2, 1, 0.3))
    expected = np.ones((data.validation.sample_count, 2))
    backend = ModelBackend(
        lambda *_: ModelResult(expected, {"units": 32}, 1.0, tmp_path / "model.keras")
    )
    result = backend.run(data, model_config(), output_config(tmp_path))
    assert np.array_equal(result.predictions, expected)


@pytest.mark.parametrize("predictions", [np.ones((1, 1)), np.full((2, 2), np.nan)])
def test_rejects_invalid_predictions(
    daily_frame: pd.DataFrame, tmp_path: Path, predictions: np.ndarray
) -> None:
    data = prepare_windows(daily_frame, WindowConfig(4, 2, 1, 0.3))
    backend = ModelBackend(
        lambda *_: ModelResult(predictions, {}, None, tmp_path / "model.keras")
    )
    with pytest.raises(ModelBackendError):
        backend.run(data, model_config(), output_config(tmp_path))


def test_wraps_unexpected_backend_failure(
    daily_frame: pd.DataFrame, tmp_path: Path
) -> None:
    data = prepare_windows(daily_frame, WindowConfig(4, 2, 1, 0.3))

    def fail(*_: object) -> ModelResult:
        raise RuntimeError("boom")

    with pytest.raises(ModelBackendError, match="Model backend failed"):
        ModelBackend(fail).run(data, model_config(), output_config(tmp_path))


def test_preserves_domain_backend_failure(
    daily_frame: pd.DataFrame, tmp_path: Path
) -> None:
    data = prepare_windows(daily_frame, WindowConfig(4, 2, 1, 0.3))

    def fail(*_: object) -> ModelResult:
        raise ModelBackendError("specific")

    with pytest.raises(ModelBackendError, match="specific"):
        ModelBackend(fail).run(data, model_config(), output_config(tmp_path))
