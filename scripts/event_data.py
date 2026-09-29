"""Shared event-table loading and sparse SOH label construction."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "modeling" / "event_features_rw9_rw11.csv"
BATTERIES = ["RW9", "RW10", "RW11"]

FEATURES = [
    "event_type_discharge",
    "duration_s",
    "current_mean_a",
    "current_abs_mean_a",
    "current_throughput_signed_ah",
    "throughput_magnitude_ah",
    "energy_signed_wh",
    "energy_magnitude_wh",
    "power_mean_signed_w",
    "power_mean_magnitude_w",
    "voltage_start",
    "voltage_end",
    "voltage_delta_signed",
    "voltage_pdf_std",
    "voltage_pdf_min",
    "voltage_pdf_q75",
    "voltage_pdf_q90",
    "voltage_pdf_max",
    "voltage_pdf_skew",
    "dq_dv_max_ahpv",
]


def load_events() -> pd.DataFrame:
    columns = [
        "battery",
        "event_type",
        "event_order_chronological",
        "label_charge_count",
        "assigned_checkpoint_cycle",
        "capacity_ah",
        "soh_percent",
        "initial_capacity_ah",
        "start_abs_time",
        "end_abs_time",
        *FEATURES,
    ]
    frame = pd.read_csv(SOURCE, usecols=columns)
    frame = frame[frame["battery"].isin(BATTERIES)].copy()
    frame = frame.sort_values(["battery", "event_order_chronological"]).reset_index(drop=True)
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    for _, indices in frame.groupby("battery").groups.items():
        block = frame.loc[indices, FEATURES]
        frame.loc[indices, FEATURES] = block.fillna(block.median(numeric_only=True)).fillna(0.0)
    frame["elapsed_s"] = frame["end_abs_time"] - frame.groupby("battery")["start_abs_time"].transform("min")
    frame["cumulative_throughput_ah"] = frame.groupby("battery")["throughput_magnitude_ah"].cumsum()
    return frame


def reference_table(frame: pd.DataFrame, battery: str) -> pd.DataFrame:
    battery_frame = frame[frame["battery"].eq(battery)].copy()
    references = (
        battery_frame.groupby("assigned_checkpoint_cycle", as_index=False)
        .agg(
            charge_count=("assigned_checkpoint_cycle", "first"),
            soh=("soh_percent", "first"),
            capacity_ah=("capacity_ah", "first"),
            max_event_order=("event_order_chronological", "max"),
        )
        .sort_values("charge_count")
    )
    initial = pd.DataFrame(
        {
            "assigned_checkpoint_cycle": [0],
            "charge_count": [0],
            "soh": [100.0],
            "capacity_ah": [float(battery_frame["initial_capacity_ah"].iloc[0])],
            "max_event_order": [int(battery_frame["event_order_chronological"].min()) - 1],
        }
    )
    references = pd.concat([initial, references], ignore_index=True)
    throughput_by_event = battery_frame.set_index("event_order_chronological")["cumulative_throughput_ah"]
    references["throughput_ah"] = 0.0
    for row in references.index[1:]:
        event_order = int(references.loc[row, "max_event_order"])
        references.loc[row, "throughput_ah"] = float(throughput_by_event.loc[event_order])
    return references


def interpolate_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = []
    for battery in BATTERIES:
        block = frame[frame["battery"].eq(battery)].copy()
        references = reference_table(frame, battery)
        ref_counts = references["charge_count"].to_numpy(float)
        ref_soh = references["soh"].to_numpy(float)
        ref_throughput = references["throughput_ah"].to_numpy(float)
        charge_count = block["label_charge_count"].to_numpy(float)
        throughput = block["cumulative_throughput_ah"].to_numpy(float)

        next_index = np.clip(np.searchsorted(ref_counts, charge_count, side="left"), 0, len(ref_counts) - 1)
        previous_index = np.maximum(next_index - 1, 0)
        count_span = ref_counts[next_index] - ref_counts[previous_index]
        throughput_span = ref_throughput[next_index] - ref_throughput[previous_index]
        count_weight = np.divide(
            charge_count - ref_counts[previous_index],
            count_span,
            out=np.zeros_like(charge_count),
            where=count_span != 0,
        )
        throughput_weight = np.divide(
            throughput - ref_throughput[previous_index],
            throughput_span,
            out=np.zeros_like(throughput),
            where=throughput_span != 0,
        )
        count_weight = np.clip(count_weight, 0.0, 1.0)
        throughput_weight = np.clip(throughput_weight, 0.0, 1.0)
        block["target_next_reference_soh"] = block["soh_percent"]
        block["target_previous_reference_soh"] = ref_soh[previous_index]
        block["target_linear_charge_count_soh"] = (
            (1.0 - count_weight) * ref_soh[previous_index] + count_weight * ref_soh[next_index]
        )
        block["target_throughput_interpolation_soh"] = (
            (1.0 - throughput_weight) * ref_soh[previous_index] + throughput_weight * ref_soh[next_index]
        )
        labeled.append(block)
    return pd.concat(labeled, ignore_index=True)
