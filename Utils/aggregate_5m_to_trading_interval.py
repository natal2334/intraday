"""Utilities for converting data to the trading interval."""
import pandas as pd
from config import MIN_INST

def aggregate_5m_to_trading_interval(prices: pd.Series, interval_minutes: int = MIN_INST) -> pd.Series:
    """
    Aggregate 5-minute prices into trading intervals.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")

    if interval_minutes % 5 != 0:
        raise ValueError("interval_minutes must be a multiple of 5")

    interval = f"{interval_minutes}min"
    aggregated = prices.resample(interval).mean()
    return aggregated