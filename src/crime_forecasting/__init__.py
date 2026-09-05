"""Daily crime-count forecasting with a tuned encoder-decoder LSTM."""

from crime_forecasting.config import AppConfig
from crime_forecasting.service import ForecastService

__all__ = ["AppConfig", "ForecastService"]
__version__ = "1.0.0"
