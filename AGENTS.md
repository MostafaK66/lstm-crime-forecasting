# Engineering contract

- Support Python 3.11 and 3.12 using a `src/` package layout.
- Keep configuration immutable and validate all external data before windowing.
- Keep TensorFlow, tuning, GPU use, plotting, and filesystem writes isolated.
- Unit tests must be deterministic and require no GPU or private dataset.
- Preserve chronological ordering and fit scalers using training samples only.
- Never use unsafe array stride tricks for supervised-window construction.
- Run Ruff, strict mypy, branch-aware pytest coverage, and compile checks before merge.
- Never commit datasets, generated plots, trained models, tuner state, or credentials.
