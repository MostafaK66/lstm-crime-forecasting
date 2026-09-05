# LSTM Crime Forecasting

A maintainable Python package for forecasting daily crime-record counts with a tuned
encoder-decoder LSTM. The repository was formerly named `5G-anomaly-detection`, but
its executable implementation has always processed calendar fields from a crime
dataset. The original telecommunications diagrams remain in `design-diagrams/` as
historical, non-executable material.

## What changed

- Uses a Python 3.11+ `src/` layout, Hatchling, immutable TOML configuration, strict
  typing, and domain-specific errors.
- Aggregates crime records into a complete daily series, explicitly inserting zero
  counts for days without records.
- Replaces unsafe NumPy stride tricks with bounds-checked window construction.
- Fits numeric and target scalers only on the training partition and safely handles
  constant series.
- Evaluates every forecast horizon against a correctly aligned persistence baseline.
- Isolates TensorFlow, KerasTuner, model persistence, CSV input, and artifact output
  behind replaceable boundaries.
- Writes tidy predictions, metrics, first-horizon residuals, and a JSON run manifest.

## Installation

Python 3.11 or 3.12 is recommended. TensorFlow supports both versions on current
platforms. Intel macOS is limited by TensorFlow to older releases and may not satisfy
this project's TensorFlow extra; use Linux, Apple silicon, or Windows/WSL2 for current
TensorFlow releases.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

For development, install `.[dev]`. To exercise only the deterministic data pipeline,
install the base package without extras.

## Data and configuration

Copy the example and adjust its paths:

```bash
cp config.example.toml config.toml
```

On Windows, use `Copy-Item config.example.toml config.toml`.

The input may be a CSV or a zip containing one CSV, as supported by pandas. It must
contain integer year, month, and day columns. Their names are configurable. Other
columns are ignored; each input row contributes one crime record to that date's
count. Generated files, datasets, models, and local `config.toml` are ignored by Git.

Key settings:

- `window.history`, `window.horizon`, and `window.step` define samples.
- `window.validation_fraction` reserves the newest samples for validation.
- `model.strategy` is `grid` or `bayesian`; `model.train = false` loads the configured
  `.keras` model instead of tuning it.
- `model.max_trials` limits both tuner strategies, which keeps even a grid run bounded.

## Usage

```bash
crime-forecast --config config.toml
# Equivalent:
python -m crime_forecasting --config config.toml
```

The output directory receives `predictions.csv`, `metrics.csv`, `residuals.csv`, and
`run.json`. The model directory receives the whole model in the recommended `.keras`
format plus temporary tuner state. Model training may use substantial CPU/GPU time;
no model or dataset is bundled or downloaded by the application.

## Architecture

```text
config.toml -> validated immutable config
            -> CSV records -> complete daily counts
            -> safe windows -> chronological split -> train-only scaling
            -> TensorFlow/KerasTuner boundary -> inverse-scaled predictions
            -> aligned evaluation -> CSV/JSON artifacts
```

`config.py`, `data.py`, `scaling.py`, `windows.py`, and `evaluation.py` form the
deterministic local core. `service.py` orchestrates injected boundaries.
`tensorflow_backend.py` is the hardware-dependent integration and imports optional
dependencies only when called.

## Development

```bash
make install
make quality
```

The quality target runs Ruff, strict mypy, branch-aware pytest coverage, and bytecode
compilation. Unit tests do not require TensorFlow, a GPU, network access, a real
dataset, or a model. Coverage measures the maintainable offline core and deliberately
excludes `tensorflow_backend.py`; CI compile-checks that integration, while a real
training run remains an environment-dependent acceptance test.

## Attribution and license

Copyright © 2026 MostafaK66. Released under the [MIT License](LICENSE). See
[NOTICE](NOTICE) for repository history and attribution.
