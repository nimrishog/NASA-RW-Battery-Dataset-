from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


ROOT = Path.cwd()
OUTPUT_ROOT = ROOT / "Analysis"
MAT_FILES = ["RW9.mat", "RW10.mat", "RW11.mat", "RW12.mat"]
DATE_FMT = "%d-%b-%Y %H:%M:%S"
DATE_ONLY_FMT = "%d-%b-%Y"


@dataclass
class StepRecord:
    mat_step_index: int
    comment: str
    step_type: str
    date: str
    date_dt: pd.Timestamp
    sample_count: int
    duration_s: float
    time_start: float
    time_end: float
    rel_time_start: float
    rel_time_end: float
    current_mean: float
    current_std: float
    voltage_start: float
    voltage_end: float
    temperature_start: float
    temperature_end: float


def as_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    return np.array([float(value)], dtype=np.float64)


def parse_step_date(raw_value: object) -> pd.Timestamp:
    text = str(raw_value)
    for fmt in (DATE_FMT, DATE_ONLY_FMT):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def load_battery(mat_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"]
    steps = data.step

    step_rows: list[dict[str, object]] = []
    charge_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []

    cycle = 0
    for mat_step_index, step in enumerate(steps, start=1):
        time_vec = as_vector(step.time)
        rel_vec = as_vector(step.relativeTime)
        voltage_vec = as_vector(step.voltage)
        current_vec = as_vector(step.current)
        temp_vec = as_vector(step.temperature)
        sample_count = int(time_vec.size)
        duration_s = float(rel_vec[-1] - rel_vec[0]) if sample_count > 1 else 0.0
        date_dt = parse_step_date(step.date)

        step_rows.append(
            {
                "mat_step_index": mat_step_index,
                "comment": str(step.comment),
                "step_type": str(step.type),
                "date": str(step.date),
                "date_dt": date_dt,
                "sample_count": sample_count,
                "duration_s": duration_s,
                "time_start": float(time_vec[0]),
                "time_end": float(time_vec[-1]),
                "rel_time_start": float(rel_vec[0]),
                "rel_time_end": float(rel_vec[-1]),
                "current_mean": float(np.mean(current_vec)),
                "current_std": float(np.std(current_vec)),
                "voltage_start": float(voltage_vec[0]),
                "voltage_end": float(voltage_vec[-1]),
                "temperature_start": float(temp_vec[0]),
                "temperature_end": float(temp_vec[-1]),
            }
        )

        if str(step.comment) == "charge (random walk)" and str(step.type) == "C" and sample_count > 1:
            cycle += 1
            dt = np.diff(rel_vec) if sample_count > 1 else np.array([], dtype=np.float64)
            charged_ah = float(-np.trapz(current_vec, rel_vec) / 3600.0)
            charge_rows.append(
                {
                    "cycle": cycle,
                    "mat_step_index": mat_step_index,
                    "date": str(step.date),
                    "date_dt": date_dt,
                    "sample_count": sample_count,
                    "duration_s": duration_s,
                    "time_start": float(time_vec[0]),
                    "time_end": float(time_vec[-1]),
                    "current_mean": float(np.mean(current_vec)),
                    "current_std": float(np.std(current_vec)),
                    "current_min": float(np.min(current_vec)),
                    "current_max": float(np.max(current_vec)),
                    "voltage_start": float(voltage_vec[0]),
                    "voltage_end": float(voltage_vec[-1]),
                    "voltage_delta": float(voltage_vec[-1] - voltage_vec[0]),
                    "temperature_start": float(temp_vec[0]),
                    "temperature_end": float(temp_vec[-1]),
                    "temperature_delta": float(temp_vec[-1] - temp_vec[0]),
                    "charged_ah": charged_ah,
                    "dt_median_s": float(np.median(dt)) if dt.size else np.nan,
                    "dt_min_s": float(np.min(dt)) if dt.size else np.nan,
                    "dt_max_s": float(np.max(dt)) if dt.size else np.nan,
                }
            )

        if str(step.comment) == "reference discharge" and str(step.type) == "D":
            capacity_ah = float(np.trapz(current_vec, rel_vec) / 3600.0)
            reference_rows.append(
                {
                    "mat_step_index": mat_step_index,
                    "date": str(step.date),
                    "date_dt": date_dt,
                    "sample_count": sample_count,
                    "duration_s": duration_s,
                    "capacity_ah": capacity_ah,
                    "time_start": float(time_vec[0]),
                    "time_end": float(time_vec[-1]),
                    "voltage_start": float(voltage_vec[0]),
                    "voltage_end": float(voltage_vec[-1]),
                }
            )

    step_df = pd.DataFrame(step_rows)
    charge_df = pd.DataFrame(charge_rows)
    reference_df = pd.DataFrame(reference_rows)

    reference_df["reference_index"] = np.arange(1, len(reference_df) + 1)
    ref_charge_counts: list[int] = []
    charge_dates = charge_df["date_dt"].to_numpy()
    for ref_date in reference_df["date_dt"]:
        ref_charge_counts.append(int(np.searchsorted(charge_dates, ref_date.to_datetime64(), side="left")))
    reference_df["charge_cycle_count_before_reference"] = ref_charge_counts

    return step_df, charge_df, reference_df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summaries(
    output_dir: Path, step_df: pd.DataFrame, charge_df: pd.DataFrame, reference_df: pd.DataFrame
) -> None:
    step_cols = [
        "mat_step_index",
        "comment",
        "step_type",
        "date",
        "sample_count",
        "duration_s",
        "time_start",
        "time_end",
        "rel_time_start",
        "rel_time_end",
        "current_mean",
        "current_std",
        "voltage_start",
        "voltage_end",
        "temperature_start",
        "temperature_end",
    ]
    charge_cols = [
        "cycle",
        "mat_step_index",
        "date",
        "sample_count",
        "duration_s",
        "time_start",
        "time_end",
        "current_mean",
        "current_std",
        "current_min",
        "current_max",
        "voltage_start",
        "voltage_end",
        "voltage_delta",
        "temperature_start",
        "temperature_end",
        "temperature_delta",
        "charged_ah",
        "dt_median_s",
        "dt_min_s",
        "dt_max_s",
    ]
    reference_cols = [
        "reference_index",
        "mat_step_index",
        "date",
        "sample_count",
        "duration_s",
        "capacity_ah",
        "charge_cycle_count_before_reference",
        "time_start",
        "time_end",
        "voltage_start",
        "voltage_end",
    ]
    step_df.to_csv(output_dir / "all_steps_summary.csv", index=False, columns=step_cols)
    charge_df.to_csv(output_dir / "charge_segments_summary.csv", index=False, columns=charge_cols)
    reference_df.to_csv(output_dir / "reference_capacity_summary.csv", index=False, columns=reference_cols)


def plot_step_type_counts(output_dir: Path, step_df: pd.DataFrame, title_prefix: str) -> None:
    counts = (
        step_df.groupby(["comment", "step_type"]).size().sort_values(ascending=False).head(12)
    )
    labels = [f"{comment} [{step_type}]" for comment, step_type in counts.index]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(counts)), counts.values, color="#2f6f8f")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Step count")
    ax.set_title(f"{title_prefix}: dominant step classes")
    fig.tight_layout()
    fig.savefig(output_dir / "01_step_type_counts.png", dpi=180)
    plt.close(fig)


def plot_timeline(output_dir: Path, step_df: pd.DataFrame, title_prefix: str) -> None:
    color_map = {"C": "#d1495b", "D": "#00798c", "R": "#edae49"}
    colors = step_df["step_type"].map(color_map).fillna("#666666")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].scatter(step_df["mat_step_index"], step_df["duration_s"], s=6, c=colors, alpha=0.6)
    axes[0].set_ylabel("Step duration (s)")
    axes[0].set_title(f"{title_prefix}: step duration and cumulative experiment time")

    elapsed_hours = (step_df["time_end"] - step_df["time_end"].iloc[0]) / 3600.0
    axes[1].plot(step_df["mat_step_index"], elapsed_hours, lw=1.2, color="#222222")
    axes[1].set_ylabel("Cumulative time (h)")
    axes[1].set_xlabel("Original MAT step index")

    fig.tight_layout()
    fig.savefig(output_dir / "02_step_duration_and_cumulative_time.png", dpi=180)
    plt.close(fig)


def plot_charge_distribution(output_dir: Path, charge_df: pd.DataFrame, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].hist(charge_df["sample_count"], bins=np.arange(0.5, 302.5, 1), color="#7f5539")
    axes[0].set_title("Charge segment sample counts")
    axes[0].set_xlabel("Samples per segment")

    axes[1].hist(charge_df["duration_s"], bins=60, color="#6a994e")
    axes[1].set_title("Charge segment duration")
    axes[1].set_xlabel("Duration (s)")

    valid_dt = charge_df["dt_median_s"].dropna()
    axes[2].hist(valid_dt, bins=50, color="#386641")
    axes[2].set_title("Median sampling interval")
    axes[2].set_xlabel("Median dt (s)")

    fig.suptitle(f"{title_prefix}: segment length and sampling characteristics")
    fig.tight_layout()
    fig.savefig(output_dir / "03_charge_segment_distribution.png", dpi=180)
    plt.close(fig)


def plot_charge_trends(output_dir: Path, charge_df: pd.DataFrame, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axes[0, 0].plot(charge_df["cycle"], charge_df["duration_s"], lw=0.8, color="#5e548e")
    axes[0, 0].set_ylabel("Duration (s)")
    axes[0, 0].set_title("Segment duration")

    axes[0, 1].plot(charge_df["cycle"], charge_df["charged_ah"], lw=0.8, color="#9f86c0")
    axes[0, 1].set_ylabel("Charged Ah")
    axes[0, 1].set_title("Estimated charged amount per segment")

    axes[1, 0].plot(charge_df["cycle"], charge_df["voltage_start"], lw=0.8, label="start", color="#1d3557")
    axes[1, 0].plot(charge_df["cycle"], charge_df["voltage_end"], lw=0.8, label="end", color="#e63946")
    axes[1, 0].set_ylabel("Voltage (V)")
    axes[1, 0].set_title("Voltage window")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(
        charge_df["cycle"],
        charge_df["temperature_start"],
        lw=0.8,
        label="start",
        color="#2a9d8f",
    )
    axes[1, 1].plot(
        charge_df["cycle"],
        charge_df["temperature_end"],
        lw=0.8,
        label="end",
        color="#e76f51",
    )
    axes[1, 1].set_ylabel("Temperature (C)")
    axes[1, 1].set_title("Temperature window")
    axes[1, 1].legend(loc="best")

    for ax in axes[1]:
        ax.set_xlabel("Charge segment index (cycle)")

    fig.suptitle(f"{title_prefix}: charge-segment trends across life")
    fig.tight_layout()
    fig.savefig(output_dir / "04_charge_segment_trends.png", dpi=180)
    plt.close(fig)


def plot_reference_capacity(output_dir: Path, reference_df: pd.DataFrame, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(reference_df["reference_index"], reference_df["capacity_ah"], marker="o", lw=1.2, ms=3)
    axes[0].set_xlabel("Reference test index")
    axes[0].set_ylabel("Capacity (Ah)")
    axes[0].set_title("Capacity fade over reference tests")

    axes[1].plot(
        reference_df["charge_cycle_count_before_reference"],
        reference_df["capacity_ah"],
        marker="o",
        lw=1.2,
        ms=3,
    )
    axes[1].set_xlabel("Charge segments completed before reference")
    axes[1].set_ylabel("Capacity (Ah)")
    axes[1].set_title("Capacity vs usable charge-segment count")

    fig.suptitle(f"{title_prefix}: reference-discharge capacity benchmarks")
    fig.tight_layout()
    fig.savefig(output_dir / "05_reference_capacity_degradation.png", dpi=180)
    plt.close(fig)


def plot_representative_segments(
    output_dir: Path, mat_path: Path, charge_df: pd.DataFrame, title_prefix: str
) -> None:
    quantile_cycles = sorted(
        {
            int(charge_df["cycle"].iloc[0]),
            int(charge_df["cycle"].quantile(0.25)),
            int(charge_df["cycle"].quantile(0.50)),
            int(charge_df["cycle"].quantile(0.75)),
            int(charge_df["cycle"].iloc[-1]),
        }
    )
    cycle_to_step = dict(zip(charge_df["cycle"], charge_df["mat_step_index"]))
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"]
    steps = data.step

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    cmap = plt.get_cmap("viridis", len(quantile_cycles))
    for idx, cycle in enumerate(quantile_cycles):
        step = steps[int(cycle_to_step[cycle]) - 1]
        rel = as_vector(step.relativeTime)
        voltage = as_vector(step.voltage)
        current = as_vector(step.current)
        temp = as_vector(step.temperature)
        label = f"cycle {cycle} (step {cycle_to_step[cycle]})"
        axes[0].plot(rel, voltage, lw=1.2, color=cmap(idx), label=label)
        axes[1].plot(rel, current, lw=1.2, color=cmap(idx), label=label)
        axes[2].plot(rel, temp, lw=1.2, color=cmap(idx), label=label)

    axes[0].set_ylabel("Voltage (V)")
    axes[1].set_ylabel("Current (A)")
    axes[2].set_ylabel("Temperature (C)")
    axes[2].set_xlabel("Relative time within charge segment (s)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{title_prefix}: representative partial-charge segments")
    fig.tight_layout()
    fig.savefig(output_dir / "06_representative_charge_segments.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    battery_name: str,
    step_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> None:
    top_types = (
        step_df.groupby(["comment", "step_type"]).size().sort_values(ascending=False).head(8)
    )
    lines = [
        f"# {battery_name} analysis",
        "",
        "## Core counts",
        f"- Total MAT steps: {len(step_df):,}",
        f"- Kept partial-charge segments: {len(charge_df):,}",
        f"- Total charge-segment samples: {int(charge_df['sample_count'].sum()):,}",
        f"- Reference discharges: {len(reference_df):,}",
        "",
        "## Charge-segment sequence shape",
        f"- Sample count range: {int(charge_df['sample_count'].min())} to {int(charge_df['sample_count'].max())}",
        f"- Median sample count: {float(charge_df['sample_count'].median()):.1f}",
        f"- Duration range (s): {charge_df['duration_s'].min():.2f} to {charge_df['duration_s'].max():.2f}",
        f"- Median duration (s): {charge_df['duration_s'].median():.2f}",
        f"- Median sampling interval (s): {charge_df['dt_median_s'].median():.3f}",
        "",
        "## Reference capacity",
        f"- Initial reference capacity (Ah): {reference_df['capacity_ah'].iloc[0]:.6f}",
        f"- Final reference capacity (Ah): {reference_df['capacity_ah'].iloc[-1]:.6f}",
        f"- Capacity fade (Ah): {reference_df['capacity_ah'].iloc[0] - reference_df['capacity_ah'].iloc[-1]:.6f}",
        "",
        "## Dominant step classes",
    ]
    for (comment, step_type), count in top_types.items():
        lines.append(f"- {comment} [{step_type}]: {count:,}")
    lines.extend(
        [
            "",
            "## Files",
            "- `all_steps_summary.csv`: one row per original MAT step.",
            "- `charge_segments_summary.csv`: one row per kept partial-charge segment.",
            "- `reference_capacity_summary.csv`: reference-discharge benchmark capacities.",
            "- `01_*.png` to `06_*.png`: decision-oriented figures for model design.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_aggregate_reference(output_dir: Path, refs_by_battery: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for battery_name, ref_df in refs_by_battery.items():
        ax.plot(
            ref_df["charge_cycle_count_before_reference"],
            ref_df["capacity_ah"],
            marker="o",
            ms=3,
            lw=1.2,
            label=battery_name,
        )
    ax.set_xlabel("Charge segments completed before reference")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Reference-discharge capacity fade across batteries")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "all_batteries_reference_capacity.png", dpi=180)
    plt.close(fig)


def plot_aggregate_distributions(output_dir: Path, charges_by_battery: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for battery_name, charge_df in charges_by_battery.items():
        axes[0].hist(charge_df["sample_count"], bins=np.arange(0.5, 302.5, 1), histtype="step", lw=1.2, label=battery_name)
        axes[1].hist(charge_df["duration_s"], bins=60, histtype="step", lw=1.2, label=battery_name)
    axes[0].set_title("Sample count per partial-charge segment")
    axes[0].set_xlabel("Samples per segment")
    axes[1].set_title("Duration per partial-charge segment")
    axes[1].set_xlabel("Duration (s)")
    for ax in axes:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "all_batteries_charge_segment_distributions.png", dpi=180)
    plt.close(fig)


def write_master_report(
    output_dir: Path, charges_by_battery: dict[str, pd.DataFrame], refs_by_battery: dict[str, pd.DataFrame]
) -> None:
    rows = []
    for battery_name, charge_df in charges_by_battery.items():
        ref_df = refs_by_battery[battery_name]
        rows.append(
            {
                "battery": battery_name,
                "charge_segments": len(charge_df),
                "charge_rows": int(charge_df["sample_count"].sum()),
                "min_samples": int(charge_df["sample_count"].min()),
                "max_samples": int(charge_df["sample_count"].max()),
                "median_samples": float(charge_df["sample_count"].median()),
                "median_duration_s": float(charge_df["duration_s"].median()),
                "median_dt_s": float(charge_df["dt_median_s"].median()),
                "reference_count": len(ref_df),
                "initial_capacity_ah": float(ref_df["capacity_ah"].iloc[0]),
                "final_capacity_ah": float(ref_df["capacity_ah"].iloc[-1]),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "battery_comparison_summary.csv", index=False)
    lines = [
        "# Aggregate battery analysis",
        "",
        "This folder contains cross-battery figures and the comparison summary.",
        "",
        "## Files",
        "- `all_batteries_reference_capacity.png`: capacity fade comparison across RW9-RW12.",
        "- `all_batteries_charge_segment_distributions.png`: sequence-length comparison for partial-charge segments.",
        "- `battery_comparison_summary.csv`: compact table for transformer design decisions.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    aggregate_dir = OUTPUT_ROOT / "aggregate"
    ensure_dir(aggregate_dir)

    charges_by_battery: dict[str, pd.DataFrame] = {}
    refs_by_battery: dict[str, pd.DataFrame] = {}

    for mat_name in MAT_FILES:
        mat_path = ROOT / mat_name
        battery_name = mat_path.stem
        output_dir = OUTPUT_ROOT / battery_name
        ensure_dir(output_dir)

        print(f"Analyzing {mat_name}...")
        step_df, charge_df, reference_df = load_battery(mat_path)
        save_summaries(output_dir, step_df, charge_df, reference_df)
        plot_step_type_counts(output_dir, step_df, battery_name)
        plot_timeline(output_dir, step_df, battery_name)
        plot_charge_distribution(output_dir, charge_df, battery_name)
        plot_charge_trends(output_dir, charge_df, battery_name)
        plot_reference_capacity(output_dir, reference_df, battery_name)
        plot_representative_segments(output_dir, mat_path, charge_df, battery_name)
        write_report(output_dir, battery_name, step_df, charge_df, reference_df)

        charges_by_battery[battery_name] = charge_df
        refs_by_battery[battery_name] = reference_df

    plot_aggregate_reference(aggregate_dir, refs_by_battery)
    plot_aggregate_distributions(aggregate_dir, charges_by_battery)
    write_master_report(aggregate_dir, charges_by_battery, refs_by_battery)
    print(f"Analysis saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
