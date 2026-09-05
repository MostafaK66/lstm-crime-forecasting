"""Scaler tests."""

from __future__ import annotations

import numpy as np
import pytest

from crime_forecasting.errors import DataValidationError
from crime_forecasting.scaling import StandardScaler


def test_round_trip_and_constant_feature() -> None:
    values = np.array([[[1.0, 4.0], [3.0, 4.0]]])
    scaler = StandardScaler.fit(values)
    assert scaler.scale[1] == 1.0
    assert np.allclose(scaler.inverse_transform(scaler.transform(values)), values)


@pytest.mark.parametrize("values", [[], [1.0], [[float("nan")]]])
def test_rejects_bad_fit_input(values: object) -> None:
    with pytest.raises(DataValidationError):
        StandardScaler.fit(values)


def test_rejects_feature_mismatch() -> None:
    scaler = StandardScaler.fit([[1.0, 2.0]])
    with pytest.raises(DataValidationError, match="feature count"):
        scaler.transform([[1.0]])
