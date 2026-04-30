from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from Utils.check_forecast import expected_intervals_per_day
from config import DEFAULT_BACKTEST_MONTHS, MIN_INST
from forecast.base import Forecaster
from scenarios.scenario_generator import ScenarioGenerator


class HistoricalResidualGenerator(ScenarioGenerator):
    """Generate scenarios from rolling historical forecast residuals."""

    def __init__(
        self,
        forecaster: Forecaster,
        lookback_months: int = DEFAULT_BACKTEST_MONTHS,
        interval_minutes: int = MIN_INST,
    ) -> None:
        self.forecaster = forecaster
        self.lookback_months = lookback_months
        self.interval_minutes = interval_minutes
        self.intervals_per_day = expected_intervals_per_day(interval_minutes)
        self.residuals_full_: pd.DataFrame | None = None
        self.weekday_slot_residuals_: dict[tuple[int, int], np.ndarray] | None = None
        self.slot_residuals_: dict[int, np.ndarray] | None = None
        self.global_residuals_: np.ndarray | None = None

    def fit(self, prices: pd.Series) -> None:
        history = self._validate_prices(prices)
        residual_frames: list[pd.DataFrame] = []

        for forecast_day in self._candidate_forecast_days(history.index):
            train_window = self._training_window(history, forecast_day)
            actual_day = self._actual_day(history, forecast_day)

            if len(train_window) == 0 or len(actual_day) != self.intervals_per_day:
                continue

            forecast = self._forecast_day(train_window)
            forecast = pd.Series(forecast.to_numpy(dtype=float),index=actual_day.index,)
            if len(forecast) != len(actual_day):
                continue

            residuals = actual_day.astype(float) - forecast.astype(float)
            residual_frame = pd.DataFrame(
                {
                    "timestamp": actual_day.index,
                    "weekday": actual_day.index.weekday,
                    "slot": self._slot_ids(actual_day.index),
                    "residual": residuals.to_numpy(dtype=float),
                }
            )
            residual_frames.append(residual_frame)

        if not residual_frames:
            raise RuntimeError("Not enough history to build rolling forecast residuals")

        residuals_full = pd.concat(residual_frames, ignore_index=True)
        self.residuals_full_ = residuals_full
        self.weekday_slot_residuals_ = self._group_weekday_slot_residuals(residuals_full)
        self.slot_residuals_ = self._group_slot_residuals(residuals_full)
        self.global_residuals_ = residuals_full["residual"].to_numpy(dtype=float)

    def generate(
        self,
        train_prices: pd.Series,
        forecast_day: pd.Timestamp,
        n_scenarios: int,
        random_state: int | None = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be at least 1")

        train_prices = self._validate_prices(train_prices)
        if (
            self.residuals_full_ is None
            or self.weekday_slot_residuals_ is None
            or self.slot_residuals_ is None
            or self.global_residuals_ is None
        ):
            raise RuntimeError("fit() must be called before generate()")

        self.forecaster.fit(train_prices)
        forecast=self.forecaster.forecast(self.intervals_per_day)
        forecast_index = self._forecast_index(forecast_day)
        forecast = pd.Series(
            forecast.to_numpy(dtype=float),
            index=forecast_index,
            name=forecast.name,
        )

        rng = np.random.default_rng(random_state)
        scenario_matrix  = np.empty((self.intervals_per_day, n_scenarios), dtype=float)

        for row_idx, timestamp in enumerate(forecast_index):
            residual_pool = self._residual_pool(timestamp)
            sampled = rng.choice(residual_pool, size=n_scenarios, replace=True)
            scenario_matrix [row_idx, :] = forecast.iloc[row_idx] + sampled*0

        scenarios = pd.DataFrame(
        scenario_matrix , index=forecast.index, columns=[f"scenario_{i}" for i in range(n_scenarios)])

        return forecast, scenarios
    
    def _candidate_forecast_days(self, index: pd.DatetimeIndex) -> list[pd.Timestamp]:
        normalized_days = pd.DatetimeIndex(index.normalize().unique()).sort_values()
        candidate_days: list[pd.Timestamp] = []

        for forecast_day in normalized_days:
            train_window = self._training_window_from_day(index, forecast_day)
            actual_index = self._actual_day_from_day(index, forecast_day)
            if len(train_window) > 0 and len(actual_index) == self.intervals_per_day:
                candidate_days.append(pd.Timestamp(forecast_day))

        return candidate_days

    def _forecast_day(self, train_prices: pd.Series) -> pd.Series:
        if len(train_prices) == 0:
            raise RuntimeError("Training window is empty")

        forecaster = deepcopy(self.forecaster)
        forecaster.fit(train_prices)
        forecast = forecaster.forecast(self.intervals_per_day)

        if len(forecast) != self.intervals_per_day:
            raise RuntimeError("Forecaster returned an unexpected horizon length")

        return forecast.sort_index()

    def _training_window(self, prices: pd.Series, forecast_day: pd.Timestamp) -> pd.Series:
        forecast_start = pd.Timestamp(forecast_day).normalize()
        window_start = forecast_start - pd.DateOffset(months=self.lookback_months)
        return prices[(prices.index >= window_start) & (prices.index < forecast_start)]

    def _actual_day(self, prices: pd.Series, forecast_day: pd.Timestamp) -> pd.Series:
        forecast_index = self._forecast_index(forecast_day)
        actual_day = prices.reindex(forecast_index)
        return actual_day.dropna()

    def _training_window_from_day(
        self,
        index: pd.DatetimeIndex,
        forecast_day: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        forecast_start = pd.Timestamp(forecast_day).normalize()
        window_start = forecast_start - pd.DateOffset(months=self.lookback_months)
        return index[(index >= window_start) & (index < forecast_start)]

    def _actual_day_from_day(
        self,
        index: pd.DatetimeIndex,
        forecast_day: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        forecast_index = self._forecast_index(forecast_day)
        mask = index.isin(forecast_index)
        return index[mask]

    def _group_weekday_slot_residuals(
        self,
        residuals_full: pd.DataFrame,
    ) -> dict[tuple[int, int], np.ndarray]:
        grouped = residuals_full.groupby(["weekday", "slot"])["residual"]
        return {
            (int(weekday), int(slot)): values.to_numpy(dtype=float)
            for (weekday, slot), values in grouped
        }

    def _group_slot_residuals(self, residuals_full: pd.DataFrame) -> dict[int, np.ndarray]:
        grouped = residuals_full.groupby("slot")["residual"]
        return {
            int(slot): values.to_numpy(dtype=float)
            for slot, values in grouped
        }

    def _residual_pool(self, timestamp: pd.Timestamp) -> np.ndarray:
        weekday_slot_key = self._weekday_slot_key(timestamp)
        residual_pool = self.weekday_slot_residuals_.get(weekday_slot_key)
        if residual_pool is not None and residual_pool.size > 0:
            return residual_pool

        slot_id = self._slot_id(timestamp)
        residual_pool = self.slot_residuals_.get(slot_id)
        if residual_pool is not None and residual_pool.size > 0:
            return residual_pool

        if self.global_residuals_.size == 0:
            raise RuntimeError("No residual history available for scenario generation")

        return self.global_residuals_

    def _forecast_index(self, forecast_day: pd.Timestamp) -> pd.DatetimeIndex:
        day_start = pd.Timestamp(forecast_day).normalize()
        return pd.date_range(
            start=day_start,
            periods=self.intervals_per_day,
            freq=f"{self.interval_minutes}min",
        )

    def _validate_prices(self, prices: pd.Series) -> pd.Series:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must have a DatetimeIndex")
        return prices.sort_index()

    def _slot_ids(self, index: pd.DatetimeIndex) -> np.ndarray:
        return ((index.hour * 60) + index.minute) // self.interval_minutes

    def _slot_id(self, timestamp: pd.Timestamp) -> int:
        return int((timestamp.hour * 60 + timestamp.minute) // self.interval_minutes)

    def _weekday_slot_key(self, timestamp: pd.Timestamp) -> tuple[int, int]:
        return (timestamp.weekday(), self._slot_id(timestamp))
