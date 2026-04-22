import numpy as np
import pandas as pd

from battery_config import (
    C_cost_holding,
    Ma_max_charge,
    Mv_max_discharge,
    Q_capacity,
    S1_storage_initial,
    alpha_storage_loss,
    grid_step,
    terminal_storage,
)
from config import MIN_INST, trading_cost
from Utils.check_forecast import expected_intervals_per_day


class LinearBatteryOptimizer:
    def __init__(self, interval_minutes: int = MIN_INST) -> None:
        self.interval_minutes = interval_minutes
        self.interval_hours = interval_minutes / 60.0
        self.max_charge = Ma_max_charge * self.interval_hours
        self.max_discharge = Mv_max_discharge * self.interval_hours
        self.name = f"linear_battery_{interval_minutes}min"
        

    def daily_pnl(
        self,
        scenarios: pd.DataFrame | pd.Series,
        realized_day: pd.Series,
    ) -> float:
        x_path = self.bellman_iters(scenarios)
        pnl = 0.0

        for t in range(len(realized_day)):
            x_t = x_path[t]
            realized = float(realized_day.iloc[t])
            pnl += -x_t * realized - trading_cost * abs(x_t)

        return pnl

    def daily_profile(self, scenarios: pd.DataFrame | pd.Series) -> dict[int, float]:
        return self.bellman_iters(scenarios)

    def g(self, s: float, x: float) -> float:
        return (1 - alpha_storage_loss) * s + x

    def nearest_state_index(self, s: float) -> int:
        return int(round(s / grid_step))

    def bellman_iters(
        self,
        scenarios: pd.DataFrame | pd.Series,
        iterations: int = 10,
    ) -> dict[int, float]:
        price_matrix = self._price_matrix(scenarios)
        n_steps = price_matrix.shape[0]

        n_s = round(Q_capacity / grid_step) + 1
        s_grid = np.linspace(0.0, Q_capacity, n_s)
        x_grid = self._action_grid()

        V = np.full((n_steps + 1, n_s), -np.inf)
        V[n_steps, self.nearest_state_index(terminal_storage)] = 0.0

        best_index: dict[tuple[int, int], int] = {}

        for t in range(n_steps - 1, -1, -1):
            expected_price = float(np.mean(price_matrix[t, :]))

            for i_s, s_t in enumerate(s_grid):
                best_value = -np.inf
                best_i = 0

                for i_x, x_t in enumerate(x_grid):
                    s_next = self.g(s_t, x_t)
                    if s_next < 0.0 or s_next > Q_capacity:
                        continue
                    closest_idx = self.nearest_state_index(s_next)
                    if not (0 <= closest_idx < n_s):
                        continue

                    value = (
                        -expected_price * x_t
                        - trading_cost * abs(x_t)
                        - C_cost_holding * s_t
                        + V[t + 1, closest_idx]
                    )

                    if value > best_value:
                        best_value = value
                        best_i = i_x

                V[t, i_s] = best_value
                best_index[t, i_s] = best_i

        x_path: dict[int, float] = {}
        storage = S1_storage_initial
        i_s = np.max(0,int(np.argmin(np.abs(s_grid - storage))))

        for t in range(n_steps):
            x_t = float(x_grid[best_index[t, i_s]])
            x_path[t] = x_t
            storage = self.g(storage, x_t)
            if storage < 0.0 or storage > Q_capacity:
                raise RuntimeError("Bellman produced an infeasible storage level")

            i_s = self.nearest_state_index(storage)
            storage = s_grid[i_s]

        return x_path

    def _action_grid(self) -> np.ndarray:
        charge = Ma_max_charge * self.interval_hours
        discharge = Mv_max_discharge * self.interval_hours
        x_grid = np.arange( self.max_discharge,
        self.max_charge + 0.5 * grid_step,
        grid_step,
        )
        x_grid = np.append(x_grid, 0.0)
        x_grid = np.unique(np.round(x_grid, 10))
        return x_grid

    def _price_matrix(self, scenarios: pd.DataFrame | pd.Series) -> np.ndarray:
        if isinstance(scenarios, pd.Series):
            prices = scenarios.sort_index().to_numpy(dtype=float).reshape(-1, 1)
        elif isinstance(scenarios, pd.DataFrame):
            prices = scenarios.sort_index().to_numpy(dtype=float)
        else:
            raise TypeError("scenarios must be a pandas Series or DataFrame")

        expected_length = expected_intervals_per_day(self.interval_minutes)
        if prices.shape[0] != expected_length:
            raise ValueError(
                f"Expected {expected_length} intervals, got {prices.shape[0]}"
            )

        return prices
