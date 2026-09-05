"""Artifact output tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crime_forecasting.artifacts import save_artifacts
from crime_forecasting.config import OutputConfig
from crime_forecasting.errors import ArtifactError
from crime_forecasting.evaluation import Evaluation


def output(tmp_path: Path) -> OutputConfig:
    return OutputConfig(
        tmp_path / "out",
        tmp_path / "models",
        "model.keras",
        "pred.csv",
        "metrics.csv",
        "residuals.csv",
        "run.json",
    )


def test_writes_all_artifacts(tmp_path: Path) -> None:
    table = pd.DataFrame({"value": [1]})
    paths = save_artifacts(
        Evaluation(table, table, table),
        output(tmp_path),
        parameters={"units": 32},
        validation_loss=0.5,
        model_path=tmp_path / "models/model.keras",
    )
    assert all(path.is_file() for path in paths)
    manifest = json.loads(paths[-1].read_text())
    assert manifest["best_parameters"] == {"units": 32}
    assert manifest["validation_loss"] == 0.5


def test_wraps_write_error(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("file")
    config = output(tmp_path)
    config = OutputConfig(
        blocked,
        config.model_directory,
        config.model_file,
        config.predictions_file,
        config.metrics_file,
        config.residuals_file,
        config.manifest_file,
    )
    table = pd.DataFrame({"value": [1]})
    with pytest.raises(ArtifactError, match="Cannot write"):
        save_artifacts(
            Evaluation(table, table, table),
            config,
            parameters={},
            validation_loss=None,
            model_path=Path("m"),
        )
