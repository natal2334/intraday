"""Base forecast interfaces and shared utilities."""
from abc import ABC, abstractmethod
import pandas as pd

class Forecaster(ABC):
    @abstractmethod
    def fit(self, prices: pd.Series) -> None:
        """Fit the forecaster on historical price data."""
        pass
    
    @abstractmethod
    def forecast(self, steps_ahead: int) -> pd.Series:
        """Forecast prices for the next `steps_ahead` periods."""
        pass