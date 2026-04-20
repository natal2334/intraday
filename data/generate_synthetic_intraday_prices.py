from __future__ import annotations

"""
Generate synthetic 5-minute electricity prices with realistic stylized facts.

The model combines:
1) deterministic seasonality:
   - intraday shape
   - weekday/weekend effect
   - annual level and annual volatility modulation
2) stochastic mean-reverting component (OU-like)
3) volatility clustering via a simple conditional volatility update
4) jumps/spikes, including occasional negative-price events
5) regime changes (normal / stressed / renewable-heavy)

Outputs:
- CSV file with timestamp and synthetic_price
- optional PNG plot of the first N days

Example:
    python generate_synthetic_intraday_prices.py \
        --start 2021-01-01 \
        --years 3 \
        --seed 42 \
        --output synthetic_prices_5m.csv \
        --plot synthetic_prices_preview.png
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Regime:
    name: str
    level_shift: float
    vol_mult: float
    jump_prob_mult: float


REGIMES = {
    "normal": Regime("normal", level_shift=0.0, vol_mult=1.0, jump_prob_mult=1.0),
    "stressed": Regime("stressed", level_shift=25.0, vol_mult=2.2, jump_prob_mult=2.0),
    "renewable": Regime("renewable", level_shift=-8.0, vol_mult=1.4, jump_prob_mult=1.6),
}


def annual_signal(day_of_year: np.ndarray) -> np.ndarray:
    """Annual level component: higher in winter, lower in summer."""
    return 10.0 * np.cos(2.0 * np.pi * (day_of_year - 15.0) / 365.25)


def annual_volatility(day_of_year: np.ndarray) -> np.ndarray:
    """Annual volatility modulation: more volatile in winter."""
    return 1.0 + 0.35 * np.cos(2.0 * np.pi * (day_of_year - 15.0) / 365.25)


def intraday_shape(hour_float: np.ndarray, month: np.ndarray) -> np.ndarray:
    """
    Intraday deterministic profile.

    Components:
    - overnight low level
    - morning ramp
    - midday solar dip (stronger in summer)
    - evening peak
    """
    # Morning ramp around 08:00-10:00
    morning = 9.0 * np.exp(-0.5 * ((hour_float - 8.5) / 1.8) ** 2)

    # Evening peak around 18:00-20:00
    evening = 15.0 * np.exp(-0.5 * ((hour_float - 18.5) / 2.1) ** 2)

    # Midday solar dip: deeper in late spring/summer
    summer_strength = 0.5 + 0.5 * np.cos(2.0 * np.pi * (month - 7.0) / 12.0)
    midday_dip = -(6.0 + 8.0 * summer_strength) * np.exp(-0.5 * ((hour_float - 13.0) / 2.2) ** 2)

    # Soft harmonic to add asymmetry and realistic curvature
    harmonic = 2.5 * np.sin(2.0 * np.pi * (hour_float - 6.0) / 24.0)

    return morning + evening + midday_dip + harmonic


def weekday_effect(dayofweek: np.ndarray) -> np.ndarray:
    """Weekends slightly lower and flatter than weekdays."""
    is_weekend = dayofweek >= 5
    return np.where(is_weekend, -5.5, 0.0)


def build_regime_series(n: int, seed: int) -> np.ndarray:
    """
    Build piecewise-constant regimes using a daily Markov chain, then upsample.
    """
    rng = np.random.default_rng(seed)
    steps_per_day = 24 * 12
    n_days = int(np.ceil(n / steps_per_day))

    states = ["normal", "stressed", "renewable"]
    idx = {s: i for i, s in enumerate(states)}

    # Transition matrix: sticky regimes, but occasional switches.
    transition = np.array(
        [
            [0.975, 0.015, 0.010],
            [0.100, 0.860, 0.040],
            [0.080, 0.030, 0.890],
        ]
    )

    current = idx["normal"]
    daily_states: list[str] = []
    for _ in range(n_days):
        daily_states.append(states[current])
        current = rng.choice(len(states), p=transition[current])

    repeated = np.repeat(np.array(daily_states, dtype=object), steps_per_day)
    return repeated[:n]


def simulate_prices(index: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)

    hour_float = index.hour + index.minute / 60.0
    day_of_year = index.dayofyear.to_numpy()
    month = index.month.to_numpy()
    dayofweek = index.dayofweek.to_numpy()

    regime_names = build_regime_series(n=n, seed=seed + 1)
    regime_level_shift = np.array([REGIMES[name].level_shift for name in regime_names])
    regime_vol_mult = np.array([REGIMES[name].vol_mult for name in regime_names])
    regime_jump_mult = np.array([REGIMES[name].jump_prob_mult for name in regime_names])

    base_level = 55.0
    mu = (
        base_level
        + annual_signal(day_of_year)
        + intraday_shape(hour_float, month)
        + weekday_effect(dayofweek)
        + regime_level_shift
    )

    # Mean-reverting deviation with conditional volatility.
    x = np.zeros(n, dtype=float)
    sigma = np.zeros(n, dtype=float)
    eps = rng.standard_normal(n)

    # Parameters chosen for 5-minute frequency.
    phi = 0.965  # mean reversion
    base_sigma = 2.4
    sigma[0] = base_sigma * annual_volatility(day_of_year[:1])[0] * regime_vol_mult[0]

    jump_component = np.zeros(n, dtype=float)
    negative_flag = np.zeros(n, dtype=int)
    spike_flag = np.zeros(n, dtype=int)

    annual_vol = annual_volatility(day_of_year)

    for t in range(1, n):
        # Simple GARCH-like volatility update with deterministic modulation.
        sigma[t] = (
            0.15 * sigma[t - 1]
            + 0.85 * base_sigma * annual_vol[t] * regime_vol_mult[t]
            + 0.10 * abs(x[t - 1])
        )

        x[t] = phi * x[t - 1] + sigma[t] * eps[t]

        # Jumps / spikes: more likely during stressed periods and around renewable ramps.
        ramp_factor = 1.0 + 0.35 * np.exp(-0.5 * ((hour_float[t] - 8.5) / 1.5) ** 2) + 0.45 * np.exp(
            -0.5 * ((hour_float[t] - 18.0) / 1.8) ** 2
        )
        renewable_factor = 1.0 + 0.50 * np.exp(-0.5 * ((hour_float[t] - 13.0) / 2.2) ** 2)

        jump_prob = 0.0012 * regime_jump_mult[t] * ramp_factor
        neg_prob = 0.0008 * regime_jump_mult[t] * renewable_factor

        u = rng.random()
        if u < neg_prob:
            # Negative-price event, more plausible around midday and renewable regime.
            jump_component[t] = -rng.uniform(25.0, 90.0)
            negative_flag[t] = 1
        elif u < neg_prob + jump_prob:
            # Positive spike.
            jump_component[t] = rng.uniform(20.0, 120.0)
            spike_flag[t] = 1

    price = mu + x + jump_component

    # Occasional price floors/ceilings are not imposed; keep realism with clipping only at extreme bounds.
    price = np.clip(price, -250.0, 1200.0)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "synthetic_price": price,
            "deterministic_level": mu,
            "mean_reverting_component": x,
            "jump_component": jump_component,
            "regime": regime_names,
            "is_negative_spike": negative_flag,
            "is_positive_spike": spike_flag,
        }
    )
    return df


def make_index(start: str, years: int) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.DateOffset(years=years)
    return pd.date_range(start=start_ts, end=end_ts, freq="5min", inclusive="left")


def save_plot(df: pd.DataFrame, output_path: Path, days: int = 14) -> None:
    import matplotlib.pyplot as plt

    points = days * 24 * 12
    preview = df.iloc[:points].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(preview["timestamp"], preview["synthetic_price"], linewidth=0.9)
    ax.set_title(f"Synthetic 5-minute electricity prices — first {days} days")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic 5-minute electricity prices.")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start timestamp, e.g. 2021-01-01")
    parser.add_argument("--years", type=int, default=3, help="Number of years to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_prices_5m.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="Optional PNG plot output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = make_index(start=args.start, years=args.years)
    df = simulate_prices(index=index, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        save_plot(df, plot_path)

    print(f"Generated {len(df):,} rows from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Saved CSV to: {output_path}")
    if args.plot:
        print(f"Saved plot to: {args.plot}")


if __name__ == "__main__":
    main()
