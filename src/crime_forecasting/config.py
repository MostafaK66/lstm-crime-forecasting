"""Immutable application configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from crime_forecasting.errors import ConfigurationError

T = TypeVar("T")


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing or invalid [{name}] section")
    return cast(dict[str, Any], value)


def _value(section: dict[str, Any], name: str, expected: type[T]) -> T:
    value = section.get(name)
    if not isinstance(value, expected):
        expected_name = expected.__name__
        raise ConfigurationError(f"'{name}' must be a {expected_name}")
    return value


def _positive(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise ConfigurationError(f"'{name}' must be {comparator}")
    return value


def _filename(value: str, name: str, *, suffix: str | None = None) -> str:
    candidate = Path(value)
    if candidate.name != value or value in {"", ".", ".."}:
        raise ConfigurationError(f"'{name}' must be a plain filename")
    if suffix is not None and candidate.suffix != suffix:
        raise ConfigurationError(f"'{name}' must end with '{suffix}'")
    return value


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Input path and date-column mapping."""

    path: Path
    year_column: str
    month_column: str
    day_column: str

    def __post_init__(self) -> None:
        columns = (self.year_column, self.month_column, self.day_column)
        if any(not column.strip() for column in columns):
            raise ConfigurationError("Date column names cannot be empty")
        if len(set(columns)) != len(columns):
            raise ConfigurationError("Date column names must be unique")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Supervised-window and chronological-split settings."""

    history: int
    horizon: int
    step: int
    validation_fraction: float

    def __post_init__(self) -> None:
        _positive(self.history, "history")
        _positive(self.horizon, "horizon")
        _positive(self.step, "step")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ConfigurationError("'validation_fraction' must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model-selection and training controls."""

    strategy: str
    train: bool
    random_seed: int
    max_trials: int
    epochs: int
    patience: int
    minimum_delta: float

    def __post_init__(self) -> None:
        if self.strategy not in {"grid", "bayesian"}:
            raise ConfigurationError("'strategy' must be 'grid' or 'bayesian'")
        _positive(self.random_seed, "random_seed", allow_zero=True)
        _positive(self.max_trials, "max_trials")
        _positive(self.epochs, "epochs")
        _positive(self.patience, "patience", allow_zero=True)
        if self.minimum_delta < 0:
            raise ConfigurationError("'minimum_delta' must be non-negative")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output locations and artifact names."""

    directory: Path
    model_directory: Path
    model_file: str
    predictions_file: str
    metrics_file: str
    residuals_file: str
    manifest_file: str

    def __post_init__(self) -> None:
        _filename(self.model_file, "model_file", suffix=".keras")
        for name, value, suffix in (
            ("predictions_file", self.predictions_file, ".csv"),
            ("metrics_file", self.metrics_file, ".csv"),
            ("residuals_file", self.residuals_file, ".csv"),
            ("manifest_file", self.manifest_file, ".json"),
        ):
            _filename(value, name, suffix=suffix)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated root configuration."""

    data: DataConfig
    window: WindowConfig
    model: ModelConfig
    output: OutputConfig

    @classmethod
    def from_toml(cls, path: Path) -> AppConfig:
        """Load a configuration file and resolve paths relative to it."""
        try:
            with path.open("rb") as stream:
                document = tomllib.load(stream)
        except OSError as error:
            message = f"Cannot read configuration '{path}': {error}"
            raise ConfigurationError(message) from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigurationError(f"Invalid TOML in '{path}': {error}") from error

        base = path.resolve().parent
        data = _section(document, "data")
        window = _section(document, "window")
        model = _section(document, "model")
        output = _section(document, "output")
        return cls(
            data=DataConfig(
                path=base / _value(data, "path", str),
                year_column=_value(data, "year_column", str),
                month_column=_value(data, "month_column", str),
                day_column=_value(data, "day_column", str),
            ),
            window=WindowConfig(
                history=_value(window, "history", int),
                horizon=_value(window, "horizon", int),
                step=_value(window, "step", int),
                validation_fraction=float(_value(window, "validation_fraction", float)),
            ),
            model=ModelConfig(
                strategy=_value(model, "strategy", str),
                train=_value(model, "train", bool),
                random_seed=_value(model, "random_seed", int),
                max_trials=_value(model, "max_trials", int),
                epochs=_value(model, "epochs", int),
                patience=_value(model, "patience", int),
                minimum_delta=float(_value(model, "minimum_delta", float)),
            ),
            output=OutputConfig(
                directory=base / _value(output, "directory", str),
                model_directory=base / _value(output, "model_directory", str),
                model_file=_value(output, "model_file", str),
                predictions_file=_value(output, "predictions_file", str),
                metrics_file=_value(output, "metrics_file", str),
                residuals_file=_value(output, "residuals_file", str),
                manifest_file=_value(output, "manifest_file", str),
            ),
        )
