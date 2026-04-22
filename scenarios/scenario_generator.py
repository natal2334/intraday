from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd


class ScenarioGenerator(ABC):
    @abstractmethod
    def fit(self, prices: pd.Series) -> None:
        """
        Learn whatever historical structure is needed for scenario generation.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        train_prices: pd.Series,
        forecast_day: pd.Timestamp,
        n_scenarios: int,
        random_state: int | None = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Return:
        - baseline forecast for the day
        - scenario matrix of shape (96, n_scenarios)
        """
        raise NotImplementedError