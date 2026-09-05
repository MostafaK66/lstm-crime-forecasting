"""CLI tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import CaptureFixture, MonkeyPatch

from crime_forecasting import cli
from crime_forecasting.errors import ConfigurationError


def test_cli_success(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    summary = SimpleNamespace(
        training_samples=3,
        validation_samples=2,
        model_path=Path("model.keras"),
        artifact_paths=(Path("metrics.csv"),),
    )
    monkeypatch.setattr(cli.AppConfig, "from_toml", lambda _: object())
    service = SimpleNamespace(run=lambda _: summary)
    monkeypatch.setattr(cli, "ForecastService", lambda: service)
    assert cli.main(["--config", "example.toml"]) == 0
    output = capsys.readouterr().out
    assert "3 training samples" in output
    assert "artifact: metrics.csv" in output


def test_cli_expected_error(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    def fail(_: Path) -> object:
        raise ConfigurationError("bad config")

    monkeypatch.setattr(cli.AppConfig, "from_toml", fail)
    assert cli.main([]) == 2
    assert "error: bad config" in capsys.readouterr().err


def test_parser_default() -> None:
    assert cli.build_parser().parse_args([]).config == Path("config.toml")
