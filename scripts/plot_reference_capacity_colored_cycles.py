from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
ANALYSIS = BASE / "Analysis"
BATTERIES = ["RW9", "RW10", "RW11", "RW12"]


def plot_battery(battery: str) -> Path:
    battery_dir = ANALYSIS / battery
    df = pd.read_csv(battery_dir / "reference_capacity_summary.csv")
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M:%S")
    df = df.sort_values("date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.plot(
        df["date"],
        df["capacity_ah"],
        color="#444444",
        linewidth=1.5,
        alpha=0.8,
        zorder=1,
    )
    scatter = ax.scatter(
        df["date"],
        df["capacity_ah"],
        c=df["charge_cycle_count_before_reference"],
        cmap="viridis",
        s=44,
        edgecolors="black",
        linewidths=0.35,
        zorder=2,
    )

    ax.set_title(f"{battery}: reference capacity over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Reference discharge capacity (Ah)")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))
    fig.autofmt_xdate(rotation=25)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Charge cycles before reference")

    out_path = battery_dir / "07_reference_capacity_over_time_colored_cycles.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    outputs = [plot_battery(battery) for battery in BATTERIES]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
