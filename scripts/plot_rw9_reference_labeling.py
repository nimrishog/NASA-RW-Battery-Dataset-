from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = BASE / "rw9" / "reference_labeling_analysis"
DATE_FMT = "%d-%b-%Y %H:%M:%S"


def as_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    return np.array([float(value)], dtype=np.float64)


def load_unique_reference_checkpoints() -> tuple[pd.DataFrame, object]:
    ref_df = pd.read_csv(BASE / "Analysis" / "RW9" / "reference_capacity_summary.csv")
    blocks = pd.read_csv(BASE / "rw9" / "capacity_blocks.csv")
    data = scipy.io.loadmat(BASE / "RW9.mat", struct_as_record=False, squeeze_me=True)["data"]
    steps = data.step

    rows: list[dict[str, object]] = []
    charge_count = 0
    rw_period_count = 0
    seen_charge_counts: set[int] = set()
    prev_charge_count = 0
    prev_rw_period_count = 0

    for mat_step_index, step in enumerate(steps, start=1):
        comment = str(step.comment)
        step_type = str(step.type)
        sample_count = as_vector(step.time).size

        if comment in {"rest (random walk)", "charge (random walk)", "discharge (random walk)"}:
            rw_period_count += 1
        if comment == "charge (random walk)" and step_type == "C" and sample_count > 1:
            charge_count += 1
        if comment == "reference discharge" and step_type == "D" and charge_count not in seen_charge_counts:
            rows.append(
                {
                    "mat_step_index": mat_step_index,
                    "reference_date": pd.to_datetime(str(step.date), format=DATE_FMT),
                    "charge_cycle_count_before_reference": charge_count,
                    "rw_period_count_before_reference": rw_period_count,
                    "charge_cycles_since_prev_reference": charge_count - prev_charge_count,
                    "rw_periods_since_prev_reference": rw_period_count - prev_rw_period_count,
                    "capacity_ah": float(np.trapz(as_vector(step.current), as_vector(step.relativeTime)) / 3600.0),
                }
            )
            seen_charge_counts.add(charge_count)
            prev_charge_count = charge_count
            prev_rw_period_count = rw_period_count

    unique_ref = pd.DataFrame(rows)
    unique_ref = unique_ref.merge(
        blocks[["capacity_block_id", "cycle_start", "cycle_end", "cycle_count", "capacity_ah_label"]],
        left_on="charge_cycle_count_before_reference",
        right_on="cycle_end",
        how="left",
    )
    unique_ref.to_csv(OUTDIR / "unique_reference_checkpoints.csv", index=False)
    blocks.to_csv(OUTDIR / "charge_capacity_blocks.csv", index=False)
    ref_df.to_csv(OUTDIR / "all_reference_discharge_steps.csv", index=False)
    return unique_ref, steps


def plot_reference_capacity(unique_ref: pd.DataFrame, ref_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].scatter(ref_df["reference_index"], ref_df["capacity_ah"], s=18, alpha=0.55, color="#9d4edd", label="all 80 reference discharge steps")
    axes[0].plot(unique_ref.index + 1, unique_ref["capacity_ah"], color="#1d3557", linewidth=1.5, marker="o", label="unique checkpoint used for labeling")
    axes[0].set_title("RW9 reference discharge capacity values")
    axes[0].set_xlabel("Reference discharge step index")
    axes[0].set_ylabel("Capacity (Ah)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(unique_ref["reference_date"], unique_ref["capacity_ah"], color="#1d3557", linewidth=1.5, marker="o")
    axes[1].set_title("Unique reference-discharge capacity over calendar time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Capacity (Ah)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.25)

    fig.savefig(OUTDIR / "01_reference_discharge_capacity.png", dpi=180)
    plt.close(fig)


def plot_label_blocks(unique_ref: pd.DataFrame, blocks: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.step(blocks["cycle_end"], blocks["capacity_ah_label"], where="post", color="#1d3557", linewidth=1.6, label="charge-block label used in pkl")
    ax.scatter(
        unique_ref["charge_cycle_count_before_reference"],
        unique_ref["capacity_ah"],
        s=30,
        color="#e76f51",
        label="reference discharge capacity checkpoint",
        zorder=3,
    )
    ax.set_title("How RW9 benchmark capacity is assigned to charge-cycle blocks")
    ax.set_xlabel("Charge cycle count")
    ax.set_ylabel("Capacity (Ah)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(OUTDIR / "02_charge_label_blocks_vs_reference.png", dpi=180)
    plt.close(fig)


def plot_block_sizes(unique_ref: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].bar(unique_ref.index + 1, unique_ref["charge_cycles_since_prev_reference"], color="#457b9d")
    axes[0].set_title("Charge cycles between unique reference-discharge checkpoints")
    axes[0].set_xlabel("Checkpoint index")
    axes[0].set_ylabel("Charge cycles in block")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(unique_ref.index + 1, unique_ref["rw_periods_since_prev_reference"], color="#2a9d8f")
    axes[1].axhline(1500, color="#d1495b", linestyle="--", linewidth=1.5, label="NASA 1500 periods")
    axes[1].axhline(3000, color="#264653", linestyle="--", linewidth=1.5, label="Observed RW MAT steps")
    axes[1].set_title("All random-walk MAT periods between unique reference checkpoints")
    axes[1].set_xlabel("Checkpoint index")
    axes[1].set_ylabel("RW MAT periods in block")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(OUTDIR / "03_block_sizes.png", dpi=180)
    plt.close(fig)


def plot_reference_step_examples(unique_ref: pd.DataFrame, steps: object) -> None:
    if len(unique_ref) < 3:
        return
    selected = [
        int(unique_ref.iloc[0]["mat_step_index"]),
        int(unique_ref.iloc[len(unique_ref) // 2]["mat_step_index"]),
        int(unique_ref.iloc[-1]["mat_step_index"]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for row, mat_step_index in enumerate(selected):
        step = steps[mat_step_index - 1]
        rel_time = as_vector(step.relativeTime)
        voltage = as_vector(step.voltage)
        current = as_vector(step.current)
        temperature = as_vector(step.temperature)

        axes[row, 0].plot(rel_time, voltage, color="#1d3557")
        axes[row, 0].set_title(f"Reference discharge step {mat_step_index}: voltage")
        axes[row, 0].set_ylabel("Voltage (V)")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(rel_time, current, color="#e76f51")
        axes[row, 1].set_title(f"Step {mat_step_index}: current")
        axes[row, 1].set_ylabel("Current (A)")
        axes[row, 1].grid(alpha=0.25)

        axes[row, 2].plot(rel_time, temperature, color="#2a9d8f")
        axes[row, 2].set_title(f"Step {mat_step_index}: temperature")
        axes[row, 2].set_ylabel("Temperature (C)")
        axes[row, 2].grid(alpha=0.25)

        for col in range(3):
            axes[row, col].set_xlabel("relTime (s)")

    fig.savefig(OUTDIR / "04_reference_discharge_step_examples.png", dpi=180)
    plt.close(fig)


def write_report(unique_ref: pd.DataFrame) -> None:
    lines = [
        "# RW9 reference-labeling analysis",
        "",
        "## What the label is",
        "- Capacity is measured at benchmark `reference discharge` steps in `RW9.mat`.",
        "- The charge-side pkl dataset does not measure capacity inside each partial charge segment.",
        "- Instead, one benchmark capacity value is assigned to a block of charge cycles until the next benchmark update.",
        "",
        "## What one label block means",
        f"- Median charge cycles per label block: {unique_ref['charge_cycles_since_prev_reference'].median():.0f}",
        f"- Median all random-walk MAT periods per checkpoint block: {unique_ref['rw_periods_since_prev_reference'].median():.0f}",
        "- In RW9, the pkl-style charge file uses one benchmark capacity value for many charge cycles, and every row belonging to those cycles inherits the same label.",
        "",
        "## About NASA's 1500 periods wording",
        "- The MAT file stores many random-walk steps, including `rest`, `charge`, and `discharge` events.",
        "- In RW9, the benchmark checkpoints appear every ~3000 random-walk MAT steps.",
        "- This is consistent with the idea that one operational 'period' on the NASA page is not the same thing as one MAT step; the MAT sequence often stores rest and active events separately.",
        "",
        "## Can you pair one benchmark label to many operational samples?",
        "- Yes. That is exactly the supervision logic used in the charge-side pkl layout.",
        "- The clean unit is one charge cycle / segment, not one independent row.",
        "- If the data are flattened into rows, then all rows belonging to that segment carry the same cycle-level capacity label.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    unique_ref, steps = load_unique_reference_checkpoints()
    ref_df = pd.read_csv(BASE / "Analysis" / "RW9" / "reference_capacity_summary.csv")
    blocks = pd.read_csv(BASE / "rw9" / "capacity_blocks.csv")

    plot_reference_capacity(unique_ref, ref_df)
    plot_label_blocks(unique_ref, blocks)
    plot_block_sizes(unique_ref)
    plot_reference_step_examples(unique_ref, steps)
    write_report(unique_ref)


if __name__ == "__main__":
    main()
