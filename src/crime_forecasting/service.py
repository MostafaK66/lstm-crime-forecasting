"""Application orchestration independent of the CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crime_forecasting.artifacts import save_artifacts
from crime_forecasting.backend import ModelBackend
from crime_forecasting.config import AppConfig, DataConfig, OutputConfig, WindowConfig
from crime_forecasting.data import load_daily_counts
from crime_forecasting.evaluation import evaluate
from crime_forecasting.windows import PreparedData, prepare_windows

DataLoader = Callable[[DataConfig], pd.DataFrame]
WindowPreparer = Callable[[pd.DataFrame, WindowConfig], PreparedData]
ArtifactWriter = Callable[..., tuple[Path, ...]]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Paths and sample counts from a completed run."""

    artifact_paths: tuple[Path, ...]
    model_path: Path
    training_samples: int
    validation_samples: int


class ForecastService:
    """Coordinate ingestion, preparation, model execution, and evaluation."""

    def __init__(
        self,
        *,
        loader: DataLoader = load_daily_counts,
        preparer: WindowPreparer = prepare_windows,
        backend: ModelBackend | None = None,
        writer: ArtifactWriter = save_artifacts,
    ) -> None:
        self._loader = loader
        self._preparer = preparer
        self._backend = backend or ModelBackend()
        self._writer = writer

    def run(self, config: AppConfig) -> RunSummary:
        """Execute one forecasting run."""
        daily = self._loader(config.data)
        prepared = self._preparer(daily, config.window)
        model_result = self._backend.run(prepared, config.model, config.output)
        actual = prepared.target_scaler.inverse_transform(prepared.validation.targets)
        evaluation = evaluate(
            actual,
            model_result.predictions,
            prepared.validation.dates,
            prepared.validation.last_observed,
        )
        paths = self._writer(
            evaluation,
            config.output,
            parameters=model_result.parameters,
            validation_loss=model_result.validation_loss,
            model_path=model_result.model_path,
        )
        return RunSummary(
            artifact_paths=paths,
            model_path=model_result.model_path,
            training_samples=prepared.train.sample_count,
            validation_samples=prepared.validation.sample_count,
        )


def model_path(output: OutputConfig) -> Path:
    """Return the configured model location."""
    return output.model_directory / output.model_file
