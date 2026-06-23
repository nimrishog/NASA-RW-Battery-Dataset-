from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
UPDATED_DIR = BASE / "charge pkl with temperature"
FEATURE_DIR = BASE / "charge engineered features"
RW_NAMES = ("RW9", "RW10", "RW11", "RW12")


@dataclass
class ChargeStep:
    cycle: int
    mat_step_index: int
    date: str
    voltage: np.ndarray
    current: np.ndarray
    time: np.ndarray
    rel_time: np.ndarray
    temperature: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.time.size)


def as_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    return np.array([float(value)], dtype=np.float64)


def robust_stat(values: np.ndarray, fn) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(fn(values))


def compute_skewness(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    mean = values.mean()
    std = values.std()
    if std < 1.0e-12:
        return 0.0
    z = (values - mean) / std
    return float(np.mean(z**3))


def compute_kurtosis(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float("nan")
    mean = values.mean()
    std = values.std()
    if std < 1.0e-12:
        return 0.0
    z = (values - mean) / std
    return float(np.mean(z**4) - 3.0)


def time_in_voltage_window(rel_time: np.ndarray, voltage: np.ndarray, lower: float, upper: float) -> float:
    dt = np.diff(rel_time)
    keep = (voltage[:-1] >= lower) & (voltage[:-1] <= upper)
    if dt.size == 0:
        return float("nan")
    return float(np.sum(np.clip(dt, 0.0, None)[keep]))


def crossing_time(rel_time: np.ndarray, voltage: np.ndarray, lower: float, upper: float) -> float:
    lower_idx = np.flatnonzero(voltage >= lower)
    upper_idx = np.flatnonzero(voltage >= upper)
    if lower_idx.size == 0 or upper_idx.size == 0:
        return float("nan")
    start = lower_idx[0]
    end_candidates = upper_idx[upper_idx >= start]
    if end_candidates.size == 0:
        return float("nan")
    end = end_candidates[0]
    return float(rel_time[end] - rel_time[start])


def load_charge_steps(mat_path: Path) -> list[ChargeStep]:
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"]
    steps = data.step

    charge_steps: list[ChargeStep] = []
    cycle = 0
    for mat_step_index, step in enumerate(steps, start=1):
        if str(step.comment) != "charge (random walk)" or str(step.type) != "C":
            continue
        time_vec = as_vector(step.time)
        if time_vec.size <= 1:
            continue
        cycle += 1
        charge_steps.append(
            ChargeStep(
                cycle=cycle,
                mat_step_index=mat_step_index,
                date=str(step.date),
                voltage=as_vector(step.voltage),
                current=as_vector(step.current),
                time=time_vec,
                rel_time=as_vector(step.relativeTime),
                temperature=as_vector(step.temperature),
            )
        )
    return charge_steps


def label_map_from_reference(battery: str, max_cycle: int) -> tuple[dict[int, float], float]:
    ref_df = pd.read_csv(BASE / "Analysis" / battery / "reference_capacity_summary.csv")
    initial_capacity = float(
        ref_df.loc[ref_df["charge_cycle_count_before_reference"].eq(0), "capacity_ah"].mean()
    )
    checkpoints = (
        ref_df.groupby("charge_cycle_count_before_reference", sort=True)["capacity_ah"]
        .mean()
        .reset_index()
        .rename(columns={"charge_cycle_count_before_reference": "cycle_count"})
    )
    checkpoints = checkpoints[checkpoints["cycle_count"] > 0].reset_index(drop=True)
    counts = checkpoints["cycle_count"].to_numpy(dtype=np.int64)
    caps = checkpoints["capacity_ah"].to_numpy(dtype=np.float64)

    label_map: dict[int, float] = {}
    for cycle in range(1, max_cycle + 1):
        idx = int(np.searchsorted(counts, cycle, side="left"))
        if idx >= len(caps):
            idx = len(caps) - 1
        label_map[cycle] = float(caps[idx])
    return label_map, initial_capacity


def verified_rw9_capacity_map(steps: list[ChargeStep]) -> tuple[dict[int, float], float]:
    df = pd.read_pickle(BASE / "rw9_datacapa.pkl")
    expected_rows = sum(step.sample_count for step in steps)
    if len(df) != expected_rows:
        raise ValueError(f"RW9 row count mismatch: {len(df)} != {expected_rows}")

    expected_cycle = np.repeat(
        np.arange(1, len(steps) + 1, dtype=np.int64),
        [step.sample_count for step in steps],
    )
    if not np.array_equal(df["cycle"].to_numpy(dtype=np.int64), expected_cycle):
        raise ValueError("RW9 cycle column does not align with MAT-derived charge steps.")

    for column_name, values in (
        ("Voltage", np.concatenate([step.voltage for step in steps])),
        ("Current", np.concatenate([step.current for step in steps])),
        ("time", np.concatenate([step.time for step in steps])),
        ("relTime", np.concatenate([step.rel_time for step in steps])),
    ):
        actual = df[column_name].to_numpy(dtype=np.float64, copy=False)
        if not np.array_equal(actual, values):
            bad = int(np.flatnonzero(actual != values)[0])
            raise ValueError(f"RW9 verified pickle mismatch in {column_name} at row {bad}")

    cap_map = (
        df.groupby("cycle", sort=True)["Capacity"]
        .first()
        .to_dict()
    )
    ref_df = pd.read_csv(BASE / "Analysis" / "RW9" / "reference_capacity_summary.csv")
    initial_capacity = float(
        ref_df.loc[ref_df["charge_cycle_count_before_reference"].eq(0), "capacity_ah"].mean()
    )
    return {int(k): float(v) for k, v in cap_map.items()}, initial_capacity


def sample_pdf_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_skew": compute_skewness(values),
        f"{prefix}_kurtosis": compute_kurtosis(values),
    }


def build_updated_dataframe(
    battery: str,
    steps: list[ChargeStep],
    capacity_map: dict[int, float],
) -> pd.DataFrame:
    frames = []
    for step in steps:
        capacity = capacity_map[step.cycle]
        frames.append(
            pd.DataFrame(
                {
                    "Voltage": step.voltage.astype(np.float64, copy=False),
                    "Current": step.current.astype(np.float64, copy=False),
                    "time": step.time.astype(np.float64, copy=False),
                    "relTime": step.rel_time.astype(np.float64, copy=False),
                    "cycle": np.full(step.sample_count, step.cycle, dtype=np.int64),
                    "Capacity": np.full(step.sample_count, capacity, dtype=np.float64),
                    "Temperature": step.temperature.astype(np.float64, copy=False),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_cycle_features(
    battery: str,
    steps: list[ChargeStep],
    capacity_map: dict[int, float],
    initial_capacity: float,
    label_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step in steps:
        voltage = step.voltage
        current = step.current
        rel_time = step.rel_time
        temperature = step.temperature

        dt = np.diff(rel_time)
        dt_clipped = np.clip(dt, 0.0, None)
        dq = -current[:-1] * dt_clipped / 3600.0
        charge_cumulative = np.r_[0.0, np.cumsum(dq)]
        power_in = voltage * (-current)

        delta_v = np.diff(voltage)
        delta_t = np.diff(rel_time)
        delta_temp = np.diff(temperature)
        dv_dt = np.divide(delta_v, delta_t, out=np.full_like(delta_v, np.nan), where=np.abs(delta_t) > 1.0e-8)
        dtemp_dt = np.divide(
            delta_temp,
            delta_t,
            out=np.full_like(delta_temp, np.nan),
            where=np.abs(delta_t) > 1.0e-8,
        )
        dq_dv = np.divide(dq, delta_v, out=np.full_like(dq, np.nan), where=np.abs(delta_v) > 1.0e-6)
        dv_dq = np.divide(delta_v, dq, out=np.full_like(delta_v, np.nan), where=np.abs(dq) > 1.0e-10)

        duration_s = float(rel_time[-1] - rel_time[0])
        charged_ah = float(np.trapz(-current, rel_time) / 3600.0)
        energy_in_wh = float(np.trapz(power_in, rel_time) / 3600.0)
        capacity_ah = float(capacity_map[step.cycle])

        row = {
            "battery": battery,
            "cycle": step.cycle,
            "mat_step_index": step.mat_step_index,
            "date": step.date,
            "sample_count": step.sample_count,
            "duration_s": duration_s,
            "time_start": float(step.time[0]),
            "time_end": float(step.time[-1]),
            "capacity_ah": capacity_ah,
            "soh_percent": float(capacity_ah / initial_capacity * 100.0),
            "label_source": label_source,
            "charge_throughput_ah": charged_ah,
            "energy_throughput_wh": energy_in_wh,
            "power_mean_w": float(np.mean(power_in)),
            "power_max_w": float(np.max(power_in)),
            "voltage_start": float(voltage[0]),
            "voltage_end": float(voltage[-1]),
            "voltage_delta": float(voltage[-1] - voltage[0]),
            "voltage_slope_vps": float((voltage[-1] - voltage[0]) / max(duration_s, 1.0e-8)),
            "dv_dt_mean_vps": robust_stat(dv_dt, np.mean),
            "dv_dt_median_vps": robust_stat(dv_dt, np.median),
            "dv_dt_max_vps": robust_stat(dv_dt, np.max),
            "current_mean_a": float(np.mean(current)),
            "current_abs_mean_a": float(np.mean(np.abs(current))),
            "current_std_a": float(np.std(current)),
            "current_min_a": float(np.min(current)),
            "current_max_a": float(np.max(current)),
            "temperature_start_c": float(temperature[0]),
            "temperature_end_c": float(temperature[-1]),
            "temperature_delta_c": float(temperature[-1] - temperature[0]),
            "temperature_mean_c": float(np.mean(temperature)),
            "temperature_std_c": float(np.std(temperature)),
            "dtemp_dt_mean_cps": robust_stat(dtemp_dt, np.mean),
            "dtemp_dt_median_cps": robust_stat(dtemp_dt, np.median),
            "time_in_vwin_3p8_4p0_s": time_in_voltage_window(rel_time, voltage, 3.8, 4.0),
            "time_in_vwin_4p0_4p1_s": time_in_voltage_window(rel_time, voltage, 4.0, 4.1),
            "time_cross_3p9_to_4p1_s": crossing_time(rel_time, voltage, 3.9, 4.1),
            "dq_dv_median_ahpv": robust_stat(dq_dv, np.median),
            "dq_dv_max_ahpv": robust_stat(dq_dv, np.max),
            "dq_dv_min_ahpv": robust_stat(dq_dv, np.min),
            "dv_dq_median_vpah": robust_stat(dv_dq, np.median),
            "dv_dq_max_vpah": robust_stat(dv_dq, np.max),
            "dv_dq_min_vpah": robust_stat(dv_dq, np.min),
        }
        row.update(sample_pdf_stats("voltage_pdf", voltage))
        row.update(sample_pdf_stats("temperature_pdf", temperature))
        rows.append(row)

    return pd.DataFrame(rows)


def export_battery(battery: str) -> dict[str, object]:
    steps = load_charge_steps(BASE / f"{battery}.mat")
    if battery == "RW9":
        capacity_map, initial_capacity = verified_rw9_capacity_map(steps)
        label_source = "rw9_verified_pickle_capacity"
    else:
        capacity_map, initial_capacity = label_map_from_reference(battery, len(steps))
        label_source = "reference_discharge_block_label"

    updated_df = build_updated_dataframe(battery, steps, capacity_map)
    feature_df = build_cycle_features(battery, steps, capacity_map, initial_capacity, label_source)

    UPDATED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    updated_df.to_pickle(UPDATED_DIR / f"{battery}_charge_temp_datacapa.pkl")
    updated_df.head(2000).to_csv(UPDATED_DIR / f"{battery}_charge_temp_preview.csv", index=False)
    feature_df.to_csv(FEATURE_DIR / f"{battery}_charge_engineered_features.csv", index=False)

    return {
        "battery": battery,
        "rows": int(len(updated_df)),
        "cycles": int(feature_df["cycle"].nunique()),
        "initial_capacity_ah": float(initial_capacity),
        "first_capacity_ah": float(feature_df["capacity_ah"].iloc[0]),
        "last_capacity_ah": float(feature_df["capacity_ah"].iloc[-1]),
        "label_source": label_source,
        "updated_pkl": str(UPDATED_DIR / f"{battery}_charge_temp_datacapa.pkl"),
        "engineered_csv": str(FEATURE_DIR / f"{battery}_charge_engineered_features.csv"),
    }


def main() -> None:
    summaries = [export_battery(battery) for battery in RW_NAMES]
    summary = {
        "notes": [
            "Updated pkl-style charge files include Temperature aligned from the original MAT source.",
            "RW9 capacity labels come from the verified rw9_datacapa.pkl file after exact cycle/row validation.",
            "RW10-RW12 capacity labels are derived from reference-discharge benchmark blocks because no verified charge-side pickle was provided for those batteries.",
            "Engineered features are cycle-level segment features derived only from partial charging data plus aligned temperature.",
        ],
        "batteries": summaries,
    }
    UPDATED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    (UPDATED_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (FEATURE_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
