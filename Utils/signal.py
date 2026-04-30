import pandas as pd

def build_no_battery_signal(predictions: pd.Series, cost_threshold: float) -> pd.Series:
    signal = pd.Series(0.0, index=predictions.index)
    signal = signal.mask(predictions > cost_threshold, -1.0)
    signal = signal.mask(predictions < -cost_threshold, 1.0)
    return signal.fillna(0.0)