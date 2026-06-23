from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_charge_discharge_weighted_transformers as base


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "charge_discharge_voltage_window_transformer"
BATTERIES = ("RW9", "RW10", "RW11")
VOLTAGE_MIN = 3.2
VOLTAGE_MAX = 4.2
VOLTAGE_BIN_WIDTH = 0.05
VOLTAGE_EDGES = np.arange(VOLTAGE_MIN, VOLTAGE_MAX + VOLTAGE_BIN_WIDTH, VOLTAGE_BIN_WIDTH)
MAX_TOKENS = len(VOLTAGE_EDGES) - 1


SPEC = base.WeightedSpec(
    name="voltage_window_h4_d80_drop020_wd5e4",
    d_model=80,
    nhead=4,
    num_layers=3,
    dim_feedforward=160,
    dropout=0.20,
    learning_rate=3.0e-4,
    weight_decay=5.0e-4,
    epochs=10,
    batch_size=512,
    max_tokens=MAX_TOKENS,
)


def _safe_slope(delta: float, duration_s: float) -> float:
    return float(delta / max(duration_s, 1.0e-8))


def _event_elapsed_days(abs_time: np.ndarray, battery_time0: float) -> np.ndarray:
    return ((abs_time - battery_time0) / 86400.0).astype(np.float32)


def build_voltage_window_sequence(event: base.RandomWalkEvent, battery_time0: float) -> np.ndarray:
    """Turn one charge/discharge event into voltage-aligned tokens.

    Each token summarizes all samples whose voltage falls inside a fixed 50 mV
    voltage bin. This makes charge and discharge segments more comparable than
    arbitrary 30-row patches.
    """
    rows: list[np.ndarray] = []
    voltage = event.voltage.astype(np.float32, copy=False)
    current = event.current.astype(np.float32, copy=False)
    temperature = event.temperature.astype(np.float32, copy=False)
    rel_time = event.rel_time.astype(np.float32, copy=False)
    abs_time = event.abs_time.astype(np.float64, copy=False)
    elapsed_days = _event_elapsed_days(abs_time, battery_time0)
    event_type_code = -1.0 if event.event_type == "charge" else 1.0

    event_duration = float(rel_time[-1] - rel_time[0]) if rel_time.size > 1 else 0.0
    current_scale = float(np.max(np.abs(current)))
    if current_scale < 1.0e-8:
        current_scale = 1.0

    for bin_idx, (low, high) in enumerate(zip(VOLTAGE_EDGES[:-1], VOLTAGE_EDGES[1:])):
        if bin_idx == MAX_TOKENS - 1:
            keep = (voltage >= low) & (voltage <= high)
        else:
            keep = (voltage >= low) & (voltage < high)
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            continue

        v = voltage[idx]
        c = current[idx]
        t = temperature[idx]
        rt = rel_time[idx]
        at_days = elapsed_days[idx]

        if idx.size > 1:
            dt = np.diff(rt)
            dt_clipped = np.clip(dt, 0.0, None)
            c_mid = c[:-1]
            v_mid = v[:-1]
            throughput_ah = float(np.sum(np.abs(c_mid) * dt_clipped) / 3600.0)
            signed_charge_ah = float(np.sum((-c_mid) * dt_clipped) / 3600.0)
            energy_wh = float(np.sum(v_mid * np.abs(c_mid) * dt_clipped) / 3600.0)
            duration_s = float(rt[-1] - rt[0])
        else:
            throughput_ah = 0.0
            signed_charge_ah = 0.0
            energy_wh = 0.0
            duration_s = 0.0

        voltage_delta = float(v[-1] - v[0])
        temp_delta = float(t[-1] - t[0])
        progress_start = float(rt[0] / max(event_duration, 1.0e-8))
        progress_end = float(rt[-1] / max(event_duration, 1.0e-8))
        bin_center = float((low + high) / 2.0)
        bin_position = float((bin_center - VOLTAGE_MIN) / (VOLTAGE_MAX - VOLTAGE_MIN))

        rows.append(
            np.array(
                [
                    event_type_code,
                    bin_position,
                    float(idx.size),
                    duration_s,
                    throughput_ah,
                    signed_charge_ah,
                    energy_wh,
                    float(v[0]),
                    float(v[-1]),
                    float(np.mean(v)),
                    voltage_delta,
                    float(np.mean(c)),
                    float(np.std(c)),
                    float(np.mean(np.abs(c))),
                    float(np.mean(c / current_scale)),
                    float(np.mean(t)),
                    temp_delta,
                    _safe_slope(voltage_delta, duration_s),
                    _safe_slope(temp_delta, duration_s),
                    float(rt[0]),
                    float(rt[-1]),
                    progress_start,
                    progress_end,
                    float(at_days[0]),
                    float(at_days[-1]),
                    float(low),
                    float(high),
                ],
                dtype=np.float32,
            )
        )

    if rows:
        return np.stack(rows, axis=0)

    fallback = base.build_weighted_patch_sequence(event, battery_time0)
    event_col = np.full((fallback.shape[0], 1), event_type_code, dtype=np.float32)
    return np.concatenate([event_col, fallback], axis=1)


def build_battery_sequences_voltage_window(
    battery: str,
    events: list[base.RandomWalkEvent],
) -> tuple[list[np.ndarray], pd.DataFrame]:
    checkpoint_counts, capacities, initial_capacity = base.load_reference_label_table(battery)
    battery_time0 = min(float(event.abs_time[0]) for event in events)
    sequences: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for event in events:
        assigned_checkpoint, target_capacity, target_soh = base.assign_next_checkpoint(
            event.label_charge_count,
            checkpoint_counts,
            capacities,
            initial_capacity,
        )
        seq = build_voltage_window_sequence(event, battery_time0)
        sequences.append(seq)
        meta_rows.append(
            {
                "battery": battery,
                "event_type": event.event_type,
                "event_order": int(event.event_order),
                "mat_step_index": int(event.mat_step_index),
                "charge_cycle_for_label": int(event.label_charge_count),
                "assigned_checkpoint_cycle": int(assigned_checkpoint),
                "sample_id": f"{battery}_{event.event_type}_{event.event_order}",
                "target_capacity_ah": float(target_capacity),
                "target_soh_percent": float(target_soh),
                "token_count": int(len(seq)),
                "raw_row_count": int(event.sample_count),
                "initial_capacity_ah": float(initial_capacity),
            }
        )

    return sequences, pd.DataFrame(meta_rows)


def save_against_baseline(new_summary: dict[str, object]) -> None:
    baseline_path = ROOT / "charge_discharge_rw11_tuning" / "tuning_summary.csv"
    baseline = pd.read_csv(baseline_path)
    best_baseline = baseline.sort_values("test_event_r2", ascending=False).iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "model": "best_patch30_charge_discharge",
                "test_event_r2": float(best_baseline["test_event_r2"]),
                "test_event_mae": float(best_baseline["test_event_mae"]),
                "test_event_rmse": float(best_baseline["test_event_rmse"]),
                "test_checkpoint_r2": float(best_baseline["test_checkpoint_r2"]),
                "test_checkpoint_mae": float(best_baseline["test_checkpoint_mae"]),
                "test_checkpoint_rmse": float(best_baseline["test_checkpoint_rmse"]),
            },
            {
                "model": "voltage_window_charge_discharge",
                "test_event_r2": float(new_summary["test_cycle_r2"]),
                "test_event_mae": float(new_summary["test_cycle_mae"]),
                "test_event_rmse": float(new_summary["test_cycle_rmse"]),
                "test_checkpoint_r2": float(new_summary["test_checkpoint_r2"]),
                "test_checkpoint_mae": float(new_summary["test_checkpoint_mae"]),
                "test_checkpoint_rmse": float(new_summary["test_checkpoint_rmse"]),
            },
        ]
    )
    comparison.to_csv(OUTDIR / "comparison_against_best_patch30.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(comparison))
    labels = comparison["model"].tolist()
    for ax, col, title, color in zip(
        axes,
        ["test_event_r2", "test_event_mae", "test_checkpoint_r2"],
        ["RW11 event R2", "RW11 event MAE", "RW11 checkpoint R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, comparison[col], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "comparison_against_best_patch30.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base.OUTDIR = OUTDIR
    base.SPEC = SPEC
    base.set_seed()

    events = {battery: base.extract_events_from_mat(battery) for battery in BATTERIES}
    sequence_cache = {
        battery: build_battery_sequences_voltage_window(battery, events[battery])
        for battery in BATTERIES
    }
    run_summary = base.run_split(
        SPEC.name,
        {"train": ("RW9", "RW10"), "test": ("RW11",)},
        sequence_cache,
    )
    save_against_baseline(run_summary)
    (OUTDIR / "voltage_window_config.json").write_text(
        json.dumps(
            {
                "voltage_min": VOLTAGE_MIN,
                "voltage_max": VOLTAGE_MAX,
                "voltage_bin_width": VOLTAGE_BIN_WIDTH,
                "max_tokens": MAX_TOKENS,
                "spec": SPEC.__dict__,
                "run_summary": run_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
