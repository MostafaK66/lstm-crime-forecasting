"""Thin command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from crime_forecasting.config import AppConfig
from crime_forecasting.errors import CrimeForecastError
from crime_forecasting.service import ForecastService


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description="Forecast daily crime counts")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML configuration path (default: config.toml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the application and translate expected failures into concise errors."""
    arguments = build_parser().parse_args(argv)
    try:
        config = AppConfig.from_toml(arguments.config)
        summary = ForecastService().run(config)
    except CrimeForecastError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"completed: {summary.training_samples} training samples, "
        f"{summary.validation_samples} validation samples"
    )
    print(f"model: {summary.model_path}")
    for path in summary.artifact_paths:
        print(f"artifact: {path}")
    return 0
