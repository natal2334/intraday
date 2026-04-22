from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_DATA_FILE, trading_cost
from data.load_data import load_prices_from_csv
from Utils.aggregate_5m_to_trading_interval import aggregate_5m_to_trading_interval
from Utils.check_forecast import expected_intervals_per_day,calibrate_moving_average
from Utils.scenario_plot import save_scenario_plot
from optimization.no_battery import NoBatteryOptimizer
from forecast.moving_average import MovingAverageForecast
from forecast.perfect_info import PerfectInfoForecaster
from scenarios.historical_residual_generator import HistoricalResidualGenerator
from optimization.linear import LinearBatteryOptimizer
from backtesting.backtest import (
    calculate_pnl_over_last_months_no_battery,
    calculate_pnl_over_last_months_battery,
)



def main() -> None:
    prices_5m = load_prices_from_csv(DEFAULT_DATA_FILE)
    prices_ti = aggregate_5m_to_trading_interval(prices_5m)
    backtest_months=3
    horizon = expected_intervals_per_day()
    if len(prices_ti) < horizon + 1:
        raise RuntimeError("Not enough data for one full trading day plus evaluation")

    train = prices_ti.iloc[:-horizon]
    realized_day = prices_ti.iloc[-horizon:]
    current_day=pd.Series([train.iloc[-1]] + realized_day.iloc[:-1].tolist(),index=realized_day.index,name="current")
    forecast_day = realized_day.index[0]

    forecast_model = MovingAverageForecast(window=12, w1=0.7, w2=0.2)
    scenario_generator = HistoricalResidualGenerator(forecast_model)
    scenario_generator.fit(train)
    forecast, scenarios = scenario_generator.generate(
        train_prices=train,
        forecast_day=forecast_day,
        n_scenarios=1,
        random_state=10)
    
    #save_scenario_plot(
    forecast_day=forecast,
    scenarios=scenarios,
    realized_day=realized_day,
    output_path="scenarios/moving_average_scenarios.png"
    
    optimizer_no = NoBatteryOptimizer(cost_threshold=trading_cost)
    optimizer_bat=LinearBatteryOptimizer()
    forecasters = [
        ("moving_average", MovingAverageForecast(window=96*10, w1=0.8, w2=0.2)),
        ("perfect_info", PerfectInfoForecaster(prices_ti)),]
    """
    for name, forecaster in forecasters:
        forecaster.fit(train)

        predicted_day = forecaster.forecast(horizon)
        pnl = optimizer.daily_pnl(predicted_day, realized_day)

        print(f"{name}: pnl = {pnl:.2f}")
    """

    """
    best_params, calibration_table = calibrate_moving_average(
    prices_ti=prices_ti,
    window_grid=[12,  48, 96,96*10,96*100],
    w1_grid=[0.0, 0.2, 0.4, 0.6, 0.8],
    w2_grid=[0.0, 0.2, 0.4, 0.6, 0.8],
    days_to_validate=1,
)

    print("Best moving-average parameters:")
    print(best_params)
    print(calibration_table.head(10))
       """
    best_params = {  "window": 96*10, "w1": 0.8,"w2": 0.2,"w3": 1.0 - 0.8 - 0.2, "mse": 0,  }
    forecasters=[
        ("moving_average",MovingAverageForecast(window=best_params["window"], w1=best_params["w1"], w2=best_params["w2"])),
          ("perfect_info", PerfectInfoForecaster(prices_ti))]        
    
    rows = []
    for name, forecaster in forecasters:
        result_no = calculate_pnl_over_last_months_no_battery(
            prices=prices_ti,
            forecaster=forecaster,
            optimizer=optimizer_no,
            months=backtest_months,
        )
        result_no["forecast"] = name
        result_no["case"] = "no_battery"
        rows.append(result_no)

        result_bat = calculate_pnl_over_last_months_battery(
            prices=prices_ti,
            forecaster=forecaster,
            optimizer=optimizer_bat,
            months=backtest_months,
            n_scenarios=5,
            random_state=1,
        )
        result_bat["forecast"] = name
        result_bat["case"] = "battery"
        rows.append(result_bat)

    results = pd.DataFrame(rows)
    results = results[["forecast", "case", "optimizer", "pnl", "pnl_days"]]

    print(results)

if __name__ == "__main__":
    main()
