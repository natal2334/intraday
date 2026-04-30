import matplotlib.pyplot as plt
import pandas as pd


def save_scenario_plot(
    forecast_day: pd.Series,
    scenarios: pd.DataFrame,
    realized_day: pd.Series,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        forecast_day.index,
        forecast_day.values,
        label="forecast",
        linewidth=2.0,
        color="black",
    )

    for column in scenarios.columns:
        ax.plot(
            scenarios.index,
            scenarios[column].values,
            linewidth=1.0,
            alpha=0.7,
            label=column,
        )

    ax.plot(
        realized_day.index,
        realized_day.values,
        label="realized",
        linewidth=2.0,
        color="red",
    )

    ax.set_title("Forecast, Scenarios, and Realized Day")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
