import pandas as pd
from copy import deepcopy
from config import trading_cost , MIN_INST, HOURS_PER_DAY
from Utils.signal import build_no_battery_signal
from Utils.check_forecast import check_forecast, expected_intervals_per_day
from scenarios.historical_residual_generator import HistoricalResidualGenerator

def count_trades(signal: pd.Series) -> int:
    """Count executions as changes in desired position."""

    aligned_signal = signal.fillna(0.0)
    position_changes = aligned_signal.ne(aligned_signal.shift(1).fillna(0.0))
    return int(position_changes.sum())
def calculate_pnl_over_last_months(
    prices: pd.Series,
    strategy,
    optimizer=None,
    months: int = 6
) -> dict:
    if optimizer is None:
        return calculate_pnl_over_last_months_no_battery(prices, strategy, months=months)
    if hasattr(optimizer, "daily_pnl"):
        return calculate_pnl_over_last_months_battery(
            prices, strategy, optimizer=optimizer, months=months
        )

    return calculate_pnl_over_last_months_battery(prices, strategy, optimizer, months=months)


def calculate_pnl_over_last_months_no_battery(
    prices: pd.Series,
    forecaster,
    optimizer=None,
    months: int = 6
) -> dict:
    prices = prices.sort_index()
    cutoff = prices.index.max() - pd.DateOffset(months=months)
    horizon = expected_intervals_per_day()

    pnl = 0.0
    pnl_days = 0

    all_days = pd.DatetimeIndex(prices.index.normalize().unique()).sort_values()

    for day in all_days:
        if day < cutoff.normalize():
            continue

        forecast_index = pd.date_range(
            start=day,
            periods=horizon,
            freq=f"{MIN_INST}min",
        )

        realized_day = prices.reindex(forecast_index).dropna()
        if len(realized_day) != horizon:
            continue

        train = prices[prices.index < day]
        if len(train) < horizon:
            continue

        model = deepcopy(forecaster)
        model.fit(train)
        predicted_day = model.forecast(horizon)
        predicted_day = pd.Series(
            predicted_day.to_numpy(dtype=float),
            index=realized_day.index,
            name="forecast",
        )

        current_values = [train.iloc[-1]] + realized_day.iloc[:-1].tolist()
        current_day = pd.Series(
            current_values,
            index=realized_day.index,
            name="current",
        )

        pnl += optimizer.daily_pnl(current_day, predicted_day, realized_day)
        pnl_days += 1

    return {
        "strategy": getattr(forecaster, "name", forecaster.__class__.__name__),
        "optimizer": optimizer.name,
        "pnl": pnl,
        "pnl_days": pnl_days,
        }

    # legacy signal-based path
    signal = strategy.build_signal(prices)
    position_change = signal - signal.shift(1).fillna(0.0)
    trading_costs = trading_cost * position_change.abs()
    pnl_series = signal * (prices.shift(-1) - prices) - trading_costs
    evaluation_pnl = pnl_series.loc[pnl_series.index >= cutoff].dropna()

    return {
        "strategy": strategy.name,
        "optimizer": "No_battery",
        "pnl": float(evaluation_pnl.sum()),
        "trades": int((signal.loc[cutoff:] != signal.loc[cutoff:].shift(1).fillna(0.0)).sum()),
        "mse": float("nan"),
        "observations": len(evaluation_pnl),
    }




def calculate_pnl_over_last_months_battery(
    prices: pd.Series,
    forecaster,
    optimizer,
    months: int = 6,
    n_scenarios: int = 5,
    random_state: int = 1,
) -> dict:
    prices = prices.sort_index()
    cutoff = prices.index.max() - pd.DateOffset(months=months)
    horizon = expected_intervals_per_day()

    pnl=0.0
    pnl_days = 0

    all_days = pd.DatetimeIndex(prices.index.normalize().unique()).sort_values()

    for day in all_days:
        if day < cutoff.normalize():
            continue

        forecast_index = pd.date_range(
            start=day,
            periods=horizon,
            freq=f"{MIN_INST}min",
        )

        realized_day = prices.reindex(forecast_index).dropna()
        if len(realized_day) != horizon:
            continue

        train = prices[prices.index < day]
        if len(train) < horizon:
            continue

        scenario_generator = HistoricalResidualGenerator(forecaster)
        scenario_generator.fit(train)

        forecast, scenarios = scenario_generator.generate(
            train_prices=train,
            forecast_day=day,
            n_scenarios=n_scenarios,
            random_state=random_state,
        )

        pnl += optimizer.daily_pnl(scenarios, realized_day)
        pnl_days += 1

    return {
        "strategy": getattr(forecaster, "name", forecaster.__class__.__name__),
        "optimizer": optimizer.name,
        "pnl": pnl,
        "pnl_days": pnl_days,}




def calculate_average_hourly_profile(
    prices: pd.Series,
    strategy,
    optimizer,
    months: int = 6,
) -> pd.DataFrame:
    prices = prices.sort_index()
    predictions = strategy.predict_next_prices(prices)
    realized = prices.shift(-1)

    aligned = pd.concat(
        [
            prices.rename("current"),
            predictions.rename("prediction"),
            realized.rename("realized"),
        ],
        axis=1,
    ).dropna()

    cutoff = prices.index.max() - pd.DateOffset(months=months)
    aligned = aligned.loc[aligned.index >= cutoff]

    rows = []
    expected_length = expected_intervals_per_day()

    for day, day_frame in aligned.groupby(aligned.index.normalize()):
        current_day = day_frame["current"].iloc[:expected_length]
        predicted_day = day_frame["prediction"].iloc[:expected_length]
        realized_day = day_frame["realized"].iloc[:expected_length]

        if len(predicted_day) < expected_length or len(realized_day) < expected_length:
            continue

        x_path = optimizer.bellman_iters(predicted_day)

        for hour in range(len(predicted_day)):
            rows.append(
                {
                    "day": day,
                    "hour": hour,
                    "x_t": x_path[hour],
                    "current": current_day.iloc[hour],
                    "prediction": predicted_day.iloc[hour],
                    "realized": realized_day.iloc[hour],
                    "strategy": strategy.name,
                    "optimizer": optimizer.name,
                }
            )

    profile_df = pd.DataFrame(rows)

    if profile_df.empty:
        return profile_df

    return (
        profile_df.groupby(["strategy", "optimizer", "hour"], as_index=False)[
            ["x_t", "current", "prediction", "realized"] ]
        .mean() )

def compare_strategies(
    prices: pd.Series,
    strategies: list,
    optimizers: list,
    months: int = 6
) -> pd.DataFrame:
    results = []
    for strategy in strategies:
        results.append(calculate_pnl_over_last_months(prices, strategy, None, months=months))
        for optimizer in optimizers:
            results.append(calculate_pnl_over_last_months(prices, strategy, optimizer, months=months))

    return pd.DataFrame(results).sort_values("pnl", ascending=False).reset_index(drop=True)

def check_no_forward_looking(
    train: pd.Series,
    forecast: pd.Series,
    realized: pd.Series,
) -> None:
    if len(train) == 0:
        raise ValueError("train is empty")

    train_end = train.index[-1]
    forecast_start = forecast.index[0]
    realized_start = realized.index[0]

    if forecast_start <= train_end:
        raise ValueError("forecast starts inside the training sample")

    if realized_start <= train_end:
        raise ValueError("realized period starts inside the training sample")

    if not forecast.index.equals(realized.index):
        raise ValueError("forecast and realized indexes must match exactly")
    

    
"""
def compare_battery_strategies(
    prices: pd.Series,
    strategies: list,
    months: int = 6
) -> pd.DataFrame:
    results = [calculate_battery_pnl_over_last_months(prices, strategy, months=months)  for strategy in strategies    ]

    return (pd.DataFrame(results)
        .sort_values("Battery_PNL", ascending=True, na_position="last")
        .reset_index(drop=True)
)

def compare_battery_strategies_mse(prices: pd.Series,
    strategies: list,
    months: int = 6
) -> pd.DataFrame:
    results = [battery_pnl_over_last_months(prices, strategy, months=months)  for strategy in strategies    ]

    return (pd.DataFrame(results)
        .sort_values("mse", ascending=True, na_position="last")
        .reset_index(drop=True)
)
"""
