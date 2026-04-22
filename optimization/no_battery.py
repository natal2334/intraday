"""No-battery optimization logic."""
import pandas as pd
from Utils import check_forecast
from Utils.check_forecast import expected_intervals_per_day
from config import MIN_INST, trading_cost
from battery_config import Ma_max_charge, Mv_max_discharge

class NoBatteryOptimizer:
    def __init__(self, cost_threshold: float = 0.0,
        interval_minutes: int = MIN_INST,) -> None:
        self.cost_threshold = cost_threshold
        self.interval_minutes = interval_minutes
        self.charge_size = Ma_max_charge * interval_minutes / 60.0
        self.discharge_size = -Mv_max_discharge * interval_minutes / 60.0
        self.name = f"no_battery_cost_{cost_threshold:g}"


    def daily_pnl(
        self,
        current_day: pd.Series,
        predicted_day: pd.Series,
        realized_day: pd.Series,
    ) -> float:
        expected_length = expected_intervals_per_day(self.interval_minutes)
        check_forecast.check_forecast(predicted_day, realized_day,expected_length=expected_length )
        if not current_day.index.equals(predicted_day.index):
            raise ValueError("current_day and predicted_day indexes must match")
        pnl = 0.0
        N = len(predicted_day)
        amount=self.interval_minutes/60

        for t in range(N):
            prediction = predicted_day.iloc[t]
            realized = realized_day.iloc[t]
            current = current_day.iloc[t]
            deviation = prediction - current
            if deviation > +self.cost_threshold:
                traded_energy = -amount  # sell
            elif deviation < -self.cost_threshold:
                traded_energy =amount  # buy
            else:
                traded_energy = 0.0  # do nothing

            pnl += -traded_energy * realized - abs(traded_energy) * self.cost_threshold
        return pnl