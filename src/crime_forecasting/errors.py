"""Domain-specific exceptions."""


class CrimeForecastError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(CrimeForecastError):
    """Configuration is missing, malformed, or inconsistent."""


class DataValidationError(CrimeForecastError):
    """Input data cannot be converted into a valid daily series."""


class InsufficientDataError(DataValidationError):
    """There are too few observations for the configured windows."""


class ModelBackendError(CrimeForecastError):
    """The optional TensorFlow model backend failed."""


class ArtifactError(CrimeForecastError):
    """A result artifact could not be written."""
