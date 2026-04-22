"""Validation utilities."""
import pandas as pd
from config import MIN_INST, HOURS_PER_DAY
from forecast.moving_average import MovingAverageForecast

def expected_intervals_per_day(interval_minutes: int = MIN_INST) -> int:
    if interval_minutes <= 0 or 60 % interval_minutes != 0:
        raise ValueError("interval_minutes must divide 60 evenly")
    return HOURS_PER_DAY * 60 // interval_minutes

def check_forecast(
    predicted: pd.Series,
    realized: pd.Series,
    expected_length: int | None = None,
) -> None:
    if not isinstance(predicted, pd.Series) or not isinstance(realized, pd.Series):
        raise TypeError("predicted and realized must be pandas Series")

    if len(predicted) != len(realized):
        raise ValueError("predicted and realized must have the same length")

    if not predicted.index.equals(realized.index):
        raise ValueError("predicted and realized indexes must align")

    if expected_length is not None and len(predicted) != expected_length:
        raise ValueError(
            f"forecast length must be {expected_length}, got {len(predicted)}"
        )
    

def calibrate_moving_average(
    prices_ti: pd.Series,
    window_grid: list[int],
    w1_grid: list[float],
    w2_grid: list[float],
    days_to_validate: int = 30,
) -> tuple[dict, pd.DataFrame]:
    horizon = expected_intervals_per_day()
    if len(prices_ti) < (days_to_validate + 1) * horizon:
        raise RuntimeError("Not enough data for rolling calibration")

    results = []
    best_mse = float("inf")
    best_params = None

    for window in window_grid:
        for w1 in w1_grid:
            for w2 in w2_grid:
                if w1 + w2 > 1.0:
                    continue

                daily_mse_values = []

                for day_idx in range(days_to_validate, 0, -1):
                    split_end = len(prices_ti) - day_idx * horizon
                    split_start = split_end - horizon

                    if split_end <= 0 or split_start < 0:
                        continue

                    train = prices_ti.iloc[:split_start]
                    realized_day = prices_ti.iloc[split_start:split_end]

                    if len(train) < window or len(realized_day) != horizon:
                        continue

                    model = MovingAverageForecast(window=window, w1=w1, w2=w2)
                    model.fit(train)
                    forecast = model.forecast(horizon)
                    forecast = pd.Series(
                        forecast.to_numpy(dtype=float),
                        index=realized_day.index,
                        name="forecast",
                    )

                    mse = float(((forecast - realized_day) ** 2).mean())
                    daily_mse_values.append(mse)

                if not daily_mse_values:
                    continue

                avg_mse = float(sum(daily_mse_values) / len(daily_mse_values))
                results.append(
                    {
                        "window": window,
                        "w1": w1,
                        "w2": w2,
                        "w3": 1.0 - w1 - w2,
                        "mse": avg_mse,
                        "days": len(daily_mse_values),
                    }
                )

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = {
                        "window": window,
                        "w1": w1,
                        "w2": w2,
                        "w3": 1.0 - w1 - w2,
                        "mse": avg_mse,
                    }

    if best_params is None:
        raise RuntimeError("Calibration failed: no valid parameter combination found")

    results_df = pd.DataFrame(results).sort_values("mse").reset_index(drop=True)
    return best_params, results_df