"""Filesystem boundary for run artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from crime_forecasting.config import OutputConfig
from crime_forecasting.errors import ArtifactError
from crime_forecasting.evaluation import Evaluation


def save_artifacts(
    evaluation: Evaluation,
    output: OutputConfig,
    *,
    parameters: Mapping[str, object],
    validation_loss: float | None,
    model_path: Path,
) -> tuple[Path, ...]:
    """Write CSV artifacts and a reproducibility manifest."""
    paths = (
        output.directory / output.predictions_file,
        output.directory / output.metrics_file,
        output.directory / output.residuals_file,
        output.directory / output.manifest_file,
    )
    try:
        output.directory.mkdir(parents=True, exist_ok=True)
        evaluation.predictions.to_csv(paths[0], index=False)
        evaluation.metrics.to_csv(paths[1], index=False)
        evaluation.residuals.to_csv(paths[2], index=False)
        manifest = {
            "best_parameters": dict(parameters),
            "model_path": str(model_path),
            "validation_loss": validation_loss,
        }
        paths[3].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactError(f"Cannot write output artifacts: {error}") from error
    return paths
