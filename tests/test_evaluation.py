"""Evaluation tests."""

from __future__ import annotations

import numpy as np
import pytest

from crime_forecasting.errors import DataValidationError
from crime_forecasting.evaluation import evaluate


def test_evaluates_each_horizon_with_aligned_baseline() -> None:
    actual = np.array([[[2.0], [3.0]], [[4.0], [8.0]]])
    predicted = np.array([[1.0, 4.0], [5.0, 6.0]])
    dates = np.array(
        [["2024-01-02", "2024-01-03"], ["2024-01-03", "2024-01-04"]],
        dtype="datetime64[D]",
    )
    result = evaluate(actual, predicted, dates, [1.0, 3.0])
    assert result.metrics["model_mse"].tolist() == [1.0, 2.5]
    assert result.metrics["persistence_mse"].tolist() == [1.0, 14.5]
    assert result.predictions.shape == (4, 6)
    assert result.residuals["residual"].tolist() == [1.0, -1.0]


@pytest.mark.parametrize(
    ("actual", "predicted", "dates", "last"),
    [
        ([1.0], [[1.0]], np.ones((1, 1), dtype="datetime64[D]"), [1.0]),
        ([[1.0]], [[1.0, 2.0]], np.ones((1, 1), dtype="datetime64[D]"), [1.0]),
        ([[1.0]], [[1.0]], np.ones((2, 1), dtype="datetime64[D]"), [1.0]),
        ([[1.0]], [[1.0]], np.ones((1, 1), dtype="datetime64[D]"), [1.0, 2.0]),
        ([[1.0]], [[1.0]], np.ones((1, 1), dtype="datetime64[D]"), [float("inf")]),
        ([[float("nan")]], [[1.0]], np.ones((1, 1), dtype="datetime64[D]"), [1.0]),
    ],
)
def test_rejects_invalid_evaluation_shapes(
    actual: object, predicted: object, dates: np.ndarray, last: object
) -> None:
    with pytest.raises(DataValidationError):
        evaluate(actual, predicted, dates, last)
