"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from crime_forecasting.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    WindowConfig,
)
from crime_forecasting.errors import ConfigurationError


def test_load_example_configuration() -> None:
    config = AppConfig.from_toml(Path("config.example.toml"))
    assert config.window.history == 21
    assert config.model.strategy == "bayesian"
    assert config.data.path.is_absolute()


@pytest.mark.parametrize("content", ["", "[data\n", "[data]\npath=1"])
def test_rejects_missing_or_malformed_configuration(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        AppConfig.from_toml(path)


def test_missing_configuration_has_context(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        AppConfig.from_toml(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 1, 1, 0.2), "history"),
        ((1, 0, 1, 0.2), "horizon"),
        ((1, 1, 0, 0.2), "step"),
        ((1, 1, 1, 1.0), "validation_fraction"),
    ],
)
def test_window_validation(arguments: tuple[int, int, int, float], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        WindowConfig(*arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ("other", True, 1, 1, 1, 0, 0.0),
        ("grid", True, -1, 1, 1, 0, 0.0),
        ("grid", True, 1, 0, 1, 0, 0.0),
        ("grid", True, 1, 1, 0, 0, 0.0),
        ("grid", True, 1, 1, 1, -1, 0.0),
        ("grid", True, 1, 1, 1, 0, -0.1),
    ],
)
def test_model_validation(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("columns", [("", "M", "D"), ("Y", "Y", "D")])
def test_data_column_validation(columns: tuple[str, str, str]) -> None:
    with pytest.raises(ConfigurationError):
        DataConfig(Path("data.csv"), *columns)


@pytest.mark.parametrize(
    ("position", "value"),
    [(0, "folder/model.keras"), (0, "model.h5"), (1, "predictions.txt"), (4, "x.csv")],
)
def test_output_filename_validation(position: int, value: str) -> None:
    files = ["model.keras", "pred.csv", "metrics.csv", "res.csv", "run.json"]
    files[position] = value
    with pytest.raises(ConfigurationError):
        OutputConfig(Path("out"), Path("model"), *files)
