"""Moving-average forecast implementation."""
from forecast.base import Forecaster
import pandas as pd
from config import MIN_INST

class PerfectInfoForecaster(Forecaster):
    def __init__(self, prices: pd.Series, interval_minutes: int = MIN_INST) -> None:
        self.prices = prices.sort_index()
        self.interval_minutes = interval_minutes
        self.last_train_time = None


    def fit(self, prices: pd.Series) -> None:
        prices = prices.sort_index()
        if len(prices) == 0:
            raise ValueError("prices must not be empty")

        self.last_train_time = prices.index[-1]

    def forecast(self, steps_ahead: int) -> pd.Series:
        if self.prices is None:
            raise RuntimeError("fit() must be called before forecast()")

        if steps_ahead <= 0:
            raise ValueError("steps_ahead must be positive")

        start = self.last_train_time + pd.Timedelta(minutes=self.interval_minutes)
        index = pd.date_range( start=start, periods=steps_ahead,   freq=f"{self.interval_minutes}min",)
        future = self.prices.reindex(index)
        if len(future) < steps_ahead:
            raise ValueError("Not enough future data for perfect forecast")

        return pd.Series(
            data=future.values,
            index=index,
            name="perfect_forecast",
        )