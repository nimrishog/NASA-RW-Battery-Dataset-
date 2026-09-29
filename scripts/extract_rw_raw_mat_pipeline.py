from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io


BASE = Path(__file__).resolve().parents[1] / "data" / "raw_mat"
DEFAULT_BATTERIES = ("RW9", "RW10", "RW11")
DATE_FORMATS = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y")
VECTOR_FIELDS = ("relativeTime", "time", "voltage", "current", "temperature")
VALID_TYPES = {"C", "D", "R"}


@dataclass
class StepVectors:
    """Normalized vectors and metadata for one MATLAB step."""

    battery_id: str
    step_index: int
    comment: str
    step_type: str
    date: str
    step_category: str
    number_of_samples: int
    duration: float
    start_time: float
    end_time: float
    relative_time: np.ndarray
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    raw_lengths: dict[str, int]


@dataclass
class ChronologyState:
    """Running counters used to build helper mapping files."""

    charge_random_walk_count: int = 0
    discharge_random_walk_count: int = 0
    reference_charge_count: int = 0
    reference_discharge_count: int = 0
    rest_count: int = 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Extract raw NASA RW battery MAT files into clean CSV exports."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE,
        help="Directory containing RW9.mat, RW10.mat, and RW11.mat",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "extracted",
        help="Directory where CSV exports will be written",
    )
    parser.add_argument(
        "--batteries",
        nargs="+",
        default=list(DEFAULT_BATTERIES),
        help="Battery IDs to process, e.g. RW9 RW10 RW11",
    )
    return parser.parse_args()


def as_float_vector(value: object) -> np.ndarray:
    """Convert MATLAB scalar/array-like value to a 1D float array."""

    if value is None:
        return np.array([], dtype=np.float64)
    if isinstance(value, np.ndarray):
        try:
            return np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return np.array([], dtype=np.float64)
    try:
        return np.array([float(value)], dtype=np.float64)
    except (TypeError, ValueError):
        return np.array([], dtype=np.float64)


def safe_text(value: object) -> str:
    """Convert a raw MATLAB field to text without raising."""

    if value is None:
        return ""
    return str(value)


def get_step_field(step: object, field_name: str) -> object | None:
    """Safely read a field from a MATLAB step struct."""

    return getattr(step, field_name, None)


def detect_step_category(comment: str, step_type: str) -> str:
    """Map comment/type pairs into stable step categories."""

    comment_lc = comment.strip().lower()
    if comment_lc == "charge (random walk)" and step_type == "C":
        return "charge_random_walk"
    if comment_lc == "discharge (random walk)" and step_type == "D":
        return "discharge_random_walk"
    if comment_lc == "reference charge" and step_type == "C":
        return "reference_charge"
    if comment_lc == "reference discharge" and step_type == "D":
        return "reference_discharge"
    if step_type == "R" or "rest" in comment_lc:
        return "rest"
    return "other"


def normalize_step_vectors(
    battery_id: str,
    step_index: int,
    step: object,
    qc_writer: csv.DictWriter,
) -> StepVectors:
    """Normalize one raw MATLAB step into aligned sample vectors and emit QC issues."""

    comment = safe_text(get_step_field(step, "comment"))
    step_type = safe_text(get_step_field(step, "type"))
    date = safe_text(get_step_field(step, "date"))
    step_category = detect_step_category(comment, step_type)

    raw_vectors = {
        "relativeTime": as_float_vector(get_step_field(step, "relativeTime")),
        "time": as_float_vector(get_step_field(step, "time")),
        "voltage": as_float_vector(get_step_field(step, "voltage")),
        "current": as_float_vector(get_step_field(step, "current")),
        "temperature": as_float_vector(get_step_field(step, "temperature")),
    }
    raw_lengths = {name: int(vec.size) for name, vec in raw_vectors.items()}

    for field_name, length in raw_lengths.items():
        if length == 0:
            qc_writer.writerow(
                {
                    "battery_id": battery_id,
                    "step_index": step_index,
                    "issue_type": f"missing_{field_name}",
                    "details": f"{field_name} is missing or empty",
                }
            )

    if not comment:
        qc_writer.writerow(
            {
                "battery_id": battery_id,
                "step_index": step_index,
                "issue_type": "missing_comment",
                "details": "comment field is missing or empty",
            }
        )
    if not step_type or step_type not in VALID_TYPES:
        qc_writer.writerow(
            {
                "battery_id": battery_id,
                "step_index": step_index,
                "issue_type": "invalid_type",
                "details": f"type={step_type!r}",
            }
        )

    present_lengths = [length for length in raw_lengths.values() if length > 0]
    if not present_lengths:
        qc_writer.writerow(
            {
                "battery_id": battery_id,
                "step_index": step_index,
                "issue_type": "empty_step",
                "details": "all numeric vectors are empty",
            }
        )
        export_length = 0
    else:
        if len(set(present_lengths)) > 1:
            qc_writer.writerow(
                {
                    "battery_id": battery_id,
                    "step_index": step_index,
                    "issue_type": "inconsistent_vector_lengths",
                    "details": json.dumps(raw_lengths, sort_keys=True),
                }
            )
        export_length = min(present_lengths)

    aligned: dict[str, np.ndarray] = {}
    for field_name, vector in raw_vectors.items():
        if export_length == 0:
            aligned[field_name] = np.array([], dtype=np.float64)
        elif vector.size == 0:
            aligned[field_name] = np.full(export_length, np.nan, dtype=np.float64)
        else:
            aligned[field_name] = vector[:export_length].astype(np.float64, copy=False)

    time_vec = aligned["time"]
    rel_time_vec = aligned["relativeTime"]

    if time_vec.size > 1 and np.any(np.diff(time_vec) < 0):
        qc_writer.writerow(
            {
                "battery_id": battery_id,
                "step_index": step_index,
                "issue_type": "non_monotonic_time",
                "details": "time decreases within the step",
            }
        )
    if rel_time_vec.size > 1 and np.any(np.diff(rel_time_vec) < 0):
        qc_writer.writerow(
            {
                "battery_id": battery_id,
                "step_index": step_index,
                "issue_type": "non_monotonic_relative_time",
                "details": "relativeTime decreases within the step",
            }
        )

    start_time = float(time_vec[0]) if time_vec.size else float("nan")
    end_time = float(time_vec[-1]) if time_vec.size else float("nan")
    if rel_time_vec.size > 1:
        duration = float(rel_time_vec[-1] - rel_time_vec[0])
    elif time_vec.size > 1:
        duration = float(time_vec[-1] - time_vec[0])
    else:
        duration = 0.0

    return StepVectors(
        battery_id=battery_id,
        step_index=step_index,
        comment=comment,
        step_type=step_type,
        date=date,
        step_category=step_category,
        number_of_samples=export_length,
        duration=duration,
        start_time=start_time,
        end_time=end_time,
        relative_time=aligned["relativeTime"],
        time=aligned["time"],
        voltage=aligned["voltage"],
        current=aligned["current"],
        temperature=aligned["temperature"],
        raw_lengths=raw_lengths,
    )


def compute_reference_discharge_capacity(step_data: StepVectors) -> float:
    """Compute benchmark discharge capacity in Ah from a reference discharge step."""

    if step_data.number_of_samples <= 1:
        return float("nan")
    if np.all(np.isnan(step_data.current)):
        return float("nan")
    integration_time = step_data.relative_time
    if integration_time.size <= 1 or np.all(np.isnan(integration_time)):
        integration_time = step_data.time
    if integration_time.size <= 1 or np.all(np.isnan(integration_time)):
        return float("nan")
    capacity = float(np.trapz(step_data.current, integration_time) / 3600.0)
    return abs(capacity)


def clean_value(value: object) -> object:
    """Normalize scalar CSV values, using blanks for NaN."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def make_row(row: dict[str, object]) -> dict[str, object]:
    """Apply CSV-safe cleaning to a row."""

    return {key: clean_value(value) for key, value in row.items()}


def write_readme(output_dir: Path) -> None:
    """Write a README that explains the raw MAT structure and exported files."""

    text = """# NASA Randomized Battery Usage Raw MAT Extraction

This folder contains a clean extraction of the raw MATLAB files:

- `RW9.mat`
- `RW10.mat`
- `RW11.mat`

## Raw MATLAB structure

Each MATLAB file contains a top-level struct called `data`.

The important field is:

- `data.step`

`data.step` is an ordered sequence of step structs. Each step may include fields such as:

- `comment`
- `type`
- `date`
- `relativeTime`
- `time`
- `voltage`
- `current`
- `temperature`

The extraction pipeline preserves that original step ordering and also exports a sample-level long table.

## Exported files

- `step_metadata.csv`
  - One row per original MATLAB step.
  - Includes battery ID, step index, comment, type, date, sample count, duration, start time, end time, and derived step category.

- `long_samples.csv`
  - One row per sample point across all steps.
  - Preserves step structure through `battery_id` and `step_index`.

- `charge_random_walk.csv`
- `discharge_random_walk.csv`
- `reference_charge.csv`
- `reference_discharge.csv`
- `rests.csv`
  - Filtered sample-level exports by step category.

- `reference_discharge_capacity.csv`
  - One row per reference discharge step.
  - `capacity_ah` is computed by integrating current over time and converting to ampere-hours.

- `step_chronology_map.csv`
  - Helper mapping file for later modeling.
  - Includes step ordering and cumulative counts of charge/discharge/reference/rest steps.

- `qc_issues.csv`
  - Data quality checks found during extraction:
    - missing fields
    - inconsistent vector lengths
    - non-monotonic time
    - empty steps
    - invalid comments/types

- `summary.json`
  - Aggregate counts and quality-check totals.

## How to use these exports later for modeling

Use `step_metadata.csv` when you need:

- step-level filtering
- chronology
- mapping between raw step IDs and comments/types

Use `long_samples.csv` when you need:

- the full sample sequence for every step
- reconstruction of original charge/discharge/rest/reference sequences

Use the filtered sample tables when you need:

- only random-walk charge samples
- only random-walk discharge samples
- only benchmark reference steps
- only rest samples

Use `reference_discharge_capacity.csv` when you need:

- benchmark capacity labels from reference discharge steps

Use `step_chronology_map.csv` when you need:

- ordering of charge/discharge/reference steps
- grouping around benchmark steps
- helper counts for later segmentation or labeling logic

## Notes

- The extraction is intentionally separate from model training.
- If a step has inconsistent vector lengths, the exported sample rows are truncated to the shortest available vector length for that step, and the issue is recorded in `qc_issues.csv`.
- Missing numeric fields are exported as blanks in CSV rows.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def open_writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, object]:
    """Open a CSV writer with a header and return both writer and handle."""

    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


def process_battery(
    mat_path: Path,
    step_writer: csv.DictWriter,
    sample_writer: csv.DictWriter,
    filter_writers: dict[str, csv.DictWriter],
    capacity_writer: csv.DictWriter,
    chronology_writer: csv.DictWriter,
    qc_writer: csv.DictWriter,
) -> dict[str, int]:
    """Process one MATLAB battery file and stream rows to the output CSV writers."""

    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"]
    steps = data.step
    battery_id = mat_path.stem
    chronology = ChronologyState()

    stats = {
        "steps": 0,
        "sample_rows": 0,
        "charge_random_walk_rows": 0,
        "discharge_random_walk_rows": 0,
        "reference_charge_rows": 0,
        "reference_discharge_rows": 0,
        "rest_rows": 0,
        "reference_discharge_steps": 0,
    }

    for step_index, step in enumerate(steps, start=1):
        step_data = normalize_step_vectors(
            battery_id=battery_id,
            step_index=step_index,
            step=step,
            qc_writer=qc_writer,
        )
        stats["steps"] += 1

        charge_before = chronology.charge_random_walk_count
        discharge_before = chronology.discharge_random_walk_count
        reference_charge_before = chronology.reference_charge_count
        reference_discharge_before = chronology.reference_discharge_count
        rest_before = chronology.rest_count
        benchmark_block_id = reference_discharge_before + 1

        charge_order = ""
        discharge_order = ""
        reference_charge_order = ""
        reference_discharge_order = ""
        rest_order = ""

        if step_data.step_category == "charge_random_walk":
            chronology.charge_random_walk_count += 1
            charge_order = chronology.charge_random_walk_count
        elif step_data.step_category == "discharge_random_walk":
            chronology.discharge_random_walk_count += 1
            discharge_order = chronology.discharge_random_walk_count
        elif step_data.step_category == "reference_charge":
            chronology.reference_charge_count += 1
            reference_charge_order = chronology.reference_charge_count
        elif step_data.step_category == "reference_discharge":
            chronology.reference_discharge_count += 1
            reference_discharge_order = chronology.reference_discharge_count
        elif step_data.step_category == "rest":
            chronology.rest_count += 1
            rest_order = chronology.rest_count

        step_writer.writerow(
            make_row(
                {
                    "battery_id": step_data.battery_id,
                    "step_index": step_data.step_index,
                    "comment": step_data.comment,
                    "type": step_data.step_type,
                    "step_category": step_data.step_category,
                    "date": step_data.date,
                    "number_of_samples": step_data.number_of_samples,
                    "duration": step_data.duration,
                    "start_time": step_data.start_time,
                    "end_time": step_data.end_time,
                }
            )
        )

        chronology_writer.writerow(
            make_row(
                {
                    "battery_id": step_data.battery_id,
                    "step_index": step_data.step_index,
                    "comment": step_data.comment,
                    "type": step_data.step_type,
                    "step_category": step_data.step_category,
                    "date": step_data.date,
                    "benchmark_block_id": benchmark_block_id,
                    "charge_random_walk_count_before_step": charge_before,
                    "discharge_random_walk_count_before_step": discharge_before,
                    "reference_charge_count_before_step": reference_charge_before,
                    "reference_discharge_count_before_step": reference_discharge_before,
                    "rest_count_before_step": rest_before,
                    "charge_random_walk_order": charge_order,
                    "discharge_random_walk_order": discharge_order,
                    "reference_charge_order": reference_charge_order,
                    "reference_discharge_order": reference_discharge_order,
                    "rest_order": rest_order,
                }
            )
        )

        if step_data.step_category == "reference_discharge":
            stats["reference_discharge_steps"] += 1
            capacity_writer.writerow(
                make_row(
                    {
                        "battery_id": step_data.battery_id,
                        "step_index": step_data.step_index,
                        "date": step_data.date,
                        "capacity_ah": compute_reference_discharge_capacity(step_data),
                    }
                )
            )

        if step_data.number_of_samples == 0:
            continue

        for sample_index in range(step_data.number_of_samples):
            sample_row = make_row(
                {
                    "battery_id": step_data.battery_id,
                    "step_index": step_data.step_index,
                    "comment": step_data.comment,
                    "type": step_data.step_type,
                    "step_category": step_data.step_category,
                    "date": step_data.date,
                    "sample_index": sample_index + 1,
                    "relativeTime": step_data.relative_time[sample_index],
                    "time": step_data.time[sample_index],
                    "voltage": step_data.voltage[sample_index],
                    "current": step_data.current[sample_index],
                    "temperature": step_data.temperature[sample_index],
                }
            )
            sample_writer.writerow(sample_row)
            stats["sample_rows"] += 1

            if step_data.step_category in filter_writers:
                filter_writers[step_data.step_category].writerow(sample_row)
                stats[f"{step_data.step_category}_rows"] += 1

    return stats


def main() -> None:
    """Run the extraction pipeline from raw MAT files to CSV exports."""

    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_readme(output_dir)

    sample_fields = [
        "battery_id",
        "step_index",
        "comment",
        "type",
        "step_category",
        "date",
        "sample_index",
        "relativeTime",
        "time",
        "voltage",
        "current",
        "temperature",
    ]
    step_fields = [
        "battery_id",
        "step_index",
        "comment",
        "type",
        "step_category",
        "date",
        "number_of_samples",
        "duration",
        "start_time",
        "end_time",
    ]
    capacity_fields = ["battery_id", "step_index", "date", "capacity_ah"]
    chronology_fields = [
        "battery_id",
        "step_index",
        "comment",
        "type",
        "step_category",
        "date",
        "benchmark_block_id",
        "charge_random_walk_count_before_step",
        "discharge_random_walk_count_before_step",
        "reference_charge_count_before_step",
        "reference_discharge_count_before_step",
        "rest_count_before_step",
        "charge_random_walk_order",
        "discharge_random_walk_order",
        "reference_charge_order",
        "reference_discharge_order",
        "rest_order",
    ]
    qc_fields = ["battery_id", "step_index", "issue_type", "details"]

    step_writer, step_handle = open_writer(output_dir / "step_metadata.csv", step_fields)
    sample_writer, sample_handle = open_writer(output_dir / "long_samples.csv", sample_fields)
    capacity_writer, capacity_handle = open_writer(
        output_dir / "reference_discharge_capacity.csv",
        capacity_fields,
    )
    chronology_writer, chronology_handle = open_writer(
        output_dir / "step_chronology_map.csv",
        chronology_fields,
    )
    qc_writer, qc_handle = open_writer(output_dir / "qc_issues.csv", qc_fields)

    filter_handles: list[object] = []
    filter_writers: dict[str, csv.DictWriter] = {}
    filter_file_map = {
        "charge_random_walk": "charge_random_walk.csv",
        "discharge_random_walk": "discharge_random_walk.csv",
        "reference_charge": "reference_charge.csv",
        "reference_discharge": "reference_discharge.csv",
        "rest": "rests.csv",
    }
    for category, filename in filter_file_map.items():
        writer, handle = open_writer(output_dir / filename, sample_fields)
        filter_writers[category] = writer
        filter_handles.append(handle)

    summary: dict[str, dict[str, int]] = {}
    try:
        for battery_id in args.batteries:
            mat_path = args.base_dir / f"{battery_id}.mat"
            if not mat_path.exists():
                raise FileNotFoundError(f"Missing MAT file: {mat_path}")
            print(f"Processing {mat_path.name} ...", flush=True)
            summary[battery_id] = process_battery(
                mat_path=mat_path,
                step_writer=step_writer,
                sample_writer=sample_writer,
                filter_writers=filter_writers,
                capacity_writer=capacity_writer,
                chronology_writer=chronology_writer,
                qc_writer=qc_writer,
            )
            print(f"Finished {battery_id}: {summary[battery_id]}", flush=True)
    finally:
        step_handle.close()
        sample_handle.close()
        capacity_handle.close()
        chronology_handle.close()
        qc_handle.close()
        for handle in filter_handles:
            handle.close()

    qc_counts: dict[str, int] = {}
    qc_path = output_dir / "qc_issues.csv"
    with qc_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qc_counts[row["issue_type"]] = qc_counts.get(row["issue_type"], 0) + 1

    summary_payload = {
        "batteries": summary,
        "total_steps": int(sum(item["steps"] for item in summary.values())),
        "total_sample_rows": int(sum(item["sample_rows"] for item in summary.values())),
        "qc_issue_counts": qc_counts,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
