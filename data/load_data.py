from __future__ import annotations

from dataclasses import dataclass
import math
import random
import pandas as pd
from config import DEFAULT_DAYS, DEFAULT_BACKTEST_MONTHS,HOURS_PER_DAY

def load_prices_from_csv(path: str = "synthetic_prices_5m.csv") -> pd.Series:
    df = pd.read_csv(path)

    df["Datetime (UTC)"] = pd.to_datetime(
        df["Datetime (UTC)"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    df = df.dropna(subset=["Datetime (UTC)", "Price (EUR/MWhe)"])
    df = df.sort_values("Datetime (UTC)")

    prices = pd.Series(
        df["Price (EUR/MWhe)"].values,
        index=df["Datetime (UTC)"],
        name="power_price",
    )

    return prices


def main() -> None:
    prices = load_prices_from_csv("synthetic_prices_5m.csv")
    print(prices)


if __name__ == "__main__":
    main()
