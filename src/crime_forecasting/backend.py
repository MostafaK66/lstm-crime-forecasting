"""Injectable model-training boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from crime_forecasting.config import ModelConfig, OutputConfig
from crime_forecasting.errors import ModelBackendError
from crime_forecasting.windows import PreparedData

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Unscaled validation predictions and model-selection details."""

    predictions: FloatArray
    parameters: Mapping[str, object]
    validation_loss: float | None
    model_path: Path


TrainingRunner = Callable[[PreparedData, ModelConfig, OutputConfig], ModelResult]


class ModelBackend:
    """Run the TensorFlow integration or an injected offline implementation."""

    def __init__(self, runner: TrainingRunner | None = None) -> None:
        self._runner = runner

    def run(
        self,
        data: PreparedData,
        model: ModelConfig,
        output: OutputConfig,
    ) -> ModelResult:
        runner = self._runner
        if runner is None:
            from crime_forecasting.tensorflow_backend import run_tensorflow

            runner = run_tensorflow
        try:
            result = runner(data, model, output)
        except ModelBackendError:
            raise
        except Exception as error:
            raise ModelBackendError(f"Model backend failed: {error}") from error
        expected = (data.validation.sample_count, data.validation.targets.shape[1])
        predictions = np.asarray(result.predictions, dtype=np.float64)
        if predictions.shape != expected:
            raise ModelBackendError(
                f"Model returned shape {predictions.shape}; expected {expected}"
            )
        if not np.isfinite(predictions).all():
            raise ModelBackendError("Model returned non-finite predictions")
        return ModelResult(
            predictions=predictions,
            parameters=result.parameters,
            validation_loss=result.validation_loss,
            model_path=result.model_path,
        )
