"""Moving-average forecast implementation."""
import pandas as pd
from .base import Forecaster
from Utils.aggregate_5m_to_trading_interval import aggregate_5m_to_trading_interval
from config import MIN_INST, HOURS_PER_DAY

class MovingAverageForecast(Forecaster):
    def __init__(self, window: int = 12,w1: float = 0.7,w2: float = 0.2,interval_minutes: int = MIN_INST,) -> None:
        """
        Args:
            window: Number of periods to use for moving average
                   (e.g., 12 periods * 5min = 60min window)
        """
        self.window = window
        self.w1 = w1
        self.w2 = w2
        self.w3 = 1.0 - w1 - w2
        self.interval_minutes = interval_minutes
        self.prices_5m = None
        self.prices_ti = None
        self.recent_ma = None
        self.same_slot_avg = None
        self.latest_ti = None
    
    def fit(self, prices: pd.Series) -> None:
        self.prices_5m = prices
        self.prices_ti = aggregate_5m_to_trading_interval(prices, interval_minutes=self.interval_minutes)
        self.recent_ma = self.prices_ti.iloc[-self.window:].mean()
        self.latest_ti = self.prices_ti.iloc[-1]
        self.same_slot_avg = self._compute_same_slot_average(self.prices_ti)

    
    def forecast(self, steps_ahead: int = 1) -> pd.Series:
        if self.recent_ma is None:
            raise RuntimeError("fit() must be called before forecast()")

        value = (
            self.w1 * self.recent_ma
            + self.w2 * self.same_slot_avg
            + self.w3 * self.latest_ti
        )

        start = self.prices_ti.index[-1] + pd.Timedelta(minutes=self.interval_minutes)
        index = pd.date_range(
            start=start,
            periods=steps_ahead,
            freq=f"{self.interval_minutes}min",
        )

        return pd.Series([value] * steps_ahead, index=index, name="forecast")

    def _compute_same_slot_average(self, prices_ti: pd.Series) -> float:
        latest_ts = prices_ti.index[-1]
        cutoff = latest_ts - pd.DateOffset(months=6)

        same_slot = prices_ti[
            (prices_ti.index < latest_ts)
            & (prices_ti.index >= cutoff)
            & (prices_ti.index.time == latest_ts.time())
        ]

        if len(same_slot) == 0:
            return self.recent_ma

        return same_slot.mean()