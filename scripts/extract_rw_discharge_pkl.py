from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = BASE / "discharge pkl version"
RW_FILES = ["RW9.mat", "RW10.mat", "RW11.mat", "RW12.mat"]


@dataclass
class StepRecord:
    cycle: int
    mat_step_index: int
    date: str
    sample_count: int
    voltage: np.ndarray
    current: np.ndarray
    time: np.ndarray
    rel_time: np.ndarray


def as_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    return np.array([float(value)], dtype=np.float64)


def load_mat_data(mat_path: Path):
    return scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"]


def extract_random_walk_steps(mat_path: Path, comment: str, step_type: str) -> list[StepRecord]:
    data = load_mat_data(mat_path)
    steps = data.step
    records: list[StepRecord] = []
    cycle = 0
    for mat_step_index, step in enumerate(steps, start=1):
        if str(step.comment) != comment or str(step.type) != step_type:
            continue
        time_vec = as_vector(step.time)
        if time_vec.size <= 1:
            continue
        cycle += 1
        records.append(
            StepRecord(
                cycle=cycle,
                mat_step_index=mat_step_index,
                date=str(step.date),
                sample_count=int(time_vec.size),
                voltage=as_vector(step.voltage),
                current=as_vector(step.current),
                time=time_vec,
                rel_time=as_vector(step.relativeTime),
            )
        )
    return records


def build_reference_checkpoints(
    mat_path: Path,
    comment: str,
    step_type: str,
) -> pd.DataFrame:
    data = load_mat_data(mat_path)
    steps = data.step

    rw_count = 0
    rows: list[dict[str, object]] = []
    seen_counts: set[int] = set()

    for mat_step_index, step in enumerate(steps, start=1):
        current_comment = str(step.comment)
        current_type = str(step.type)
        sample_count = as_vector(step.time).size
        if current_comment == comment and current_type == step_type and sample_count > 1:
            rw_count += 1
            continue

        if current_comment == "reference discharge" and current_type == "D":
            rel_time = as_vector(step.relativeTime)
            current = as_vector(step.current)
            capacity_ah = float(np.trapz(current, rel_time) / 3600.0)
            if rw_count not in seen_counts:
                rows.append(
                    {
                        "rw_cycle_count_before_reference": rw_count,
                        "reference_mat_step_index": mat_step_index,
                        "reference_date": str(step.date),
                        "reference_capacity_ah": capacity_ah,
                        "reference_sample_count": int(rel_time.size),
                    }
                )
                seen_counts.add(rw_count)

    if not rows:
        raise ValueError(f"No reference discharge checkpoints found for {mat_path.name}.")
    return pd.DataFrame(rows)


def assign_capacity_labels(step_df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = checkpoint_df.sort_values("rw_cycle_count_before_reference").reset_index(drop=True)
    counts = checkpoints["rw_cycle_count_before_reference"].to_numpy(dtype=np.int64)
    caps = checkpoints["reference_capacity_ah"].to_numpy(dtype=np.float64)
    ref_steps = checkpoints["reference_mat_step_index"].to_numpy(dtype=np.int64)
    ref_dates = checkpoints["reference_date"].to_numpy(dtype=object)

    assigned_capacity: list[float] = []
    assigned_ref_step: list[int] = []
    assigned_ref_date: list[str] = []

    for cycle in step_df["cycle"].to_numpy(dtype=np.int64):
        idx = int(np.searchsorted(counts, cycle, side="left"))
        if idx >= len(counts):
            idx = len(counts) - 1
        assigned_capacity.append(float(caps[idx]))
        assigned_ref_step.append(int(ref_steps[idx]))
        assigned_ref_date.append(str(ref_dates[idx]))

    labeled = step_df.copy()
    labeled["capacity_label_ah"] = assigned_capacity
    labeled["capacity_reference_mat_step_index"] = assigned_ref_step
    labeled["capacity_reference_date"] = assigned_ref_date
    return labeled


def flatten_steps(step_df: pd.DataFrame, steps: list[StepRecord]) -> pd.DataFrame:
    capacity_by_cycle = step_df.set_index("cycle")["capacity_label_ah"].to_dict()
    rows: list[pd.DataFrame] = []
    for step in steps:
        capacity = float(capacity_by_cycle[step.cycle])
        rows.append(
            pd.DataFrame(
                {
                    "Voltage": step.voltage.astype(np.float64, copy=False),
                    "Current": step.current.astype(np.float64, copy=False),
                    "time": step.time.astype(np.float64, copy=False),
                    "relTime": step.rel_time.astype(np.float64, copy=False),
                    "cycle": np.full(step.sample_count, step.cycle, dtype=np.int64),
                    "Capacity": np.full(step.sample_count, capacity, dtype=np.float64),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_cycle_map(steps: list[StepRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle": [step.cycle for step in steps],
            "mat_step_index": [step.mat_step_index for step in steps],
            "date": [step.date for step in steps],
            "sample_count": [step.sample_count for step in steps],
            "time_start": [float(step.time[0]) for step in steps],
            "time_end": [float(step.time[-1]) for step in steps],
            "rel_time_end": [float(step.rel_time[-1]) for step in steps],
        }
    )


def export_discharge_version(mat_path: Path) -> dict[str, object]:
    steps = extract_random_walk_steps(mat_path, "discharge (random walk)", "D")
    if not steps:
        raise ValueError(f"No discharge (random walk) steps found in {mat_path.name}.")

    cycle_map = build_cycle_map(steps)
    checkpoints = build_reference_checkpoints(mat_path, "discharge (random walk)", "D")
    cycle_map = assign_capacity_labels(cycle_map, checkpoints)
    frame = flatten_steps(cycle_map, steps)

    stem = mat_path.stem.upper()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pkl_path = OUTDIR / f"{stem}_discharge_datacapa.pkl"
    cycle_map_path = OUTDIR / f"{stem}_discharge_cycle_map.csv"
    checkpoints_path = OUTDIR / f"{stem}_discharge_reference_checkpoints.csv"

    frame.to_pickle(pkl_path)
    cycle_map.to_csv(cycle_map_path, index=False)
    checkpoints.to_csv(checkpoints_path, index=False)

    return {
        "battery": stem,
        "output_pkl": str(pkl_path),
        "output_cycle_map": str(cycle_map_path),
        "output_checkpoints": str(checkpoints_path),
        "cycle_count": int(cycle_map["cycle"].nunique()),
        "row_count": int(len(frame)),
        "first_capacity_label_ah": float(cycle_map["capacity_label_ah"].iloc[0]),
        "last_capacity_label_ah": float(cycle_map["capacity_label_ah"].iloc[-1]),
        "checkpoint_count": int(len(checkpoints)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract discharge-only pkl-style random-walk datasets from RW MAT files."
    )
    parser.add_argument(
        "--batteries",
        nargs="*",
        default=["RW9"],
        help="Battery names without extension, e.g. RW9 RW10",
    )
    args = parser.parse_args()

    requested = {name.upper() for name in args.batteries}
    summaries: list[dict[str, object]] = []
    for filename in RW_FILES:
        stem = Path(filename).stem.upper()
        if stem not in requested:
            continue
        summaries.append(export_discharge_version(BASE / filename))

    if not summaries:
        raise ValueError("No matching batteries selected.")

    summary_path = OUTDIR / "discharge_extraction_summary.json"
    summary_text = {
        "notes": [
            "Rows are filtered from MAT steps where comment='discharge (random walk)', type='D', and sample_count > 1.",
            "The Capacity column is derived from the first reference discharge checkpoint at or after each discharge cycle count.",
            "These discharge labels are derived from MAT benchmark data. They are not copied from an external verified discharge pickle.",
        ],
        "batteries": summaries,
    }
    summary_path.write_text(json.dumps(summary_text, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
