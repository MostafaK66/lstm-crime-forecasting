"""Small NumPy scaler with constant-feature protection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crime_forecasting.errors import DataValidationError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StandardScaler:
    """Immutable standard scaler fitted across all axes except the final feature axis."""

    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, values: npt.ArrayLike) -> StandardScaler:
        array = _validated(values)
        axes = tuple(range(array.ndim - 1))
        mean = np.mean(array, axis=axes)
        standard_deviation = np.std(array, axis=axes)
        scale = np.where(standard_deviation == 0.0, 1.0, standard_deviation)
        return cls(mean=np.asarray(mean), scale=np.asarray(scale))

    def transform(self, values: npt.ArrayLike) -> FloatArray:
        array = _validated(values)
        _check_features(array, self.mean)
        return np.asarray((array - self.mean) / self.scale)

    def inverse_transform(self, values: npt.ArrayLike) -> FloatArray:
        array = _validated(values)
        _check_features(array, self.mean)
        return np.asarray(array * self.scale + self.mean)


def _validated(values: npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] == 0 or array.size == 0:
        raise DataValidationError("Scaler input must be a non-empty array with features")
    if not np.isfinite(array).all():
        raise DataValidationError("Scaler input contains a non-finite value")
    return array


def _check_features(values: FloatArray, parameters: FloatArray) -> None:
    if values.shape[-1] != parameters.shape[-1]:
        raise DataValidationError("Scaler input feature count does not match fitted data")
