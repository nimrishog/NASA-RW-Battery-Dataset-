from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_full_event_token_transformers import (
    EventTransformer,
    apply_standardization,
    build_patch_sequence,
    compute_standardization,
    destandardize,
    parameter_count,
    regression_metrics,
    standardize_target,
)


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
ANALYSIS_DIR = BASE / "Analysis"
OUTDIR = BASE / "charge_discharge_weighted_transformer"
EXTRACT_DIR = OUTDIR / "extracted_segments"
BATTERIES = ("RW9", "RW10", "RW11", "RW12")
PATCH_SIZE = 30


@dataclass(frozen=True)
class WeightedSpec:
    name: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    max_tokens: int


SPEC = WeightedSpec(
    name="charge_discharge_patch30_weighted",
    d_model=80,
    nhead=8,
    num_layers=3,
    dim_feedforward=160,
    dropout=0.10,
    learning_rate=3.0e-4,
    weight_decay=1.0e-4,
    epochs=10,
    batch_size=512,
    max_tokens=11,
)


SPLITS = {
    "rw9_rw10_rw11_to_rw12": {
        "train": ("RW9", "RW10", "RW11"),
        "test": ("RW12",),
    },
    "rw9_rw10_to_rw11": {
        "train": ("RW9", "RW10"),
        "test": ("RW11",),
    },
}


@dataclass
class RandomWalkEvent:
    battery: str
    event_type: str
    event_order: int
    mat_step_index: int
    date: str
    label_charge_count: int
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    rel_time: np.ndarray
    abs_time: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.abs_time.size)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    return np.array([float(value)], dtype=np.float64)


def safe_step_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return str(value.reshape(-1)[0])
    return str(value)


def minmax01(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1.0e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def maxabs01(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    if scale < 1.0e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values / scale).astype(np.float32)


def load_reference_label_table(battery: str) -> tuple[np.ndarray, np.ndarray, float]:
    ref = pd.read_csv(ANALYSIS_DIR / battery / "reference_capacity_summary.csv")
    grouped = (
        ref.groupby("charge_cycle_count_before_reference", as_index=False)
        .agg(capacity_ah=("capacity_ah", "mean"))
        .sort_values("charge_cycle_count_before_reference")
        .reset_index(drop=True)
    )
    initial_capacity = float(
        grouped.loc[grouped["charge_cycle_count_before_reference"].eq(0), "capacity_ah"].mean()
    )
    grouped = grouped[grouped["charge_cycle_count_before_reference"] > 0].reset_index(drop=True)
    checkpoint_counts = grouped["charge_cycle_count_before_reference"].to_numpy(dtype=np.int64)
    capacities = grouped["capacity_ah"].to_numpy(dtype=np.float64)
    return checkpoint_counts, capacities, initial_capacity


def assign_next_checkpoint(
    label_charge_count: int,
    checkpoint_counts: np.ndarray,
    capacities: np.ndarray,
    initial_capacity: float,
) -> tuple[int, float, float]:
    idx = int(np.searchsorted(checkpoint_counts, label_charge_count, side="left"))
    if idx >= len(checkpoint_counts):
        idx = len(checkpoint_counts) - 1
    capacity = float(capacities[idx])
    checkpoint = int(checkpoint_counts[idx])
    soh = float(capacity / initial_capacity * 100.0)
    return checkpoint, capacity, soh


def extract_events_from_mat(battery: str) -> list[RandomWalkEvent]:
    mat_path = BASE / f"{battery}.mat"
    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"]
    steps = data.step
    charge_count = 0
    discharge_count = 0
    events: list[RandomWalkEvent] = []

    for mat_step_index, step in enumerate(steps, start=1):
        comment = safe_step_text(step.comment)
        step_type = safe_step_text(step.type)
        if comment not in {"charge (random walk)", "discharge (random walk)"}:
            continue
        if step_type not in {"C", "D"}:
            continue

        time_vec = as_vector(step.time)
        if time_vec.size <= 1:
            continue

        voltage = as_vector(step.voltage)
        current = as_vector(step.current)
        temperature = as_vector(step.temperature)
        rel_time = as_vector(step.relativeTime)
        lengths = {time_vec.size, voltage.size, current.size, temperature.size, rel_time.size}
        if len(lengths) != 1:
            continue

        if comment == "charge (random walk)":
            charge_count += 1
            event_type = "charge"
            event_order = charge_count
            label_charge_count = charge_count
        else:
            discharge_count += 1
            event_type = "discharge"
            event_order = discharge_count
            label_charge_count = charge_count

        events.append(
            RandomWalkEvent(
                battery=battery,
                event_type=event_type,
                event_order=event_order,
                mat_step_index=mat_step_index,
                date=safe_step_text(step.date),
                label_charge_count=label_charge_count,
                voltage=voltage,
                current=current,
                temperature=temperature,
                rel_time=rel_time,
                abs_time=time_vec,
            )
        )
    return events


def build_segment_dataframe(events: list[RandomWalkEvent], event_type: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for event in events:
        if event.event_type != event_type:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "battery": event.battery,
                    "event_type": event.event_type,
                    "event_order": event.event_order,
                    "mat_step_index": event.mat_step_index,
                    "date": event.date,
                    "sample_index": np.arange(1, event.sample_count + 1, dtype=np.int32),
                    "relativeTime": event.rel_time,
                    "time": event.abs_time,
                    "voltage": event.voltage,
                    "current": event.current,
                    "temperature": event.temperature,
                    "label_charge_count": event.label_charge_count,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export_extracted_segments(all_events: dict[str, list[RandomWalkEvent]], overwrite: bool = False) -> pd.DataFrame:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []
    for battery, events in all_events.items():
        for event_type in ("charge", "discharge"):
            out = EXTRACT_DIR / f"{battery}_{event_type}_random_walk_segments.pkl"
            if overwrite or not out.exists():
                build_segment_dataframe(events, event_type).to_pickle(out)
        for event in events:
            metadata_rows.append(
                {
                    "battery": battery,
                    "event_type": event.event_type,
                    "event_order": event.event_order,
                    "mat_step_index": event.mat_step_index,
                    "date": event.date,
                    "sample_count": event.sample_count,
                    "duration_s": float(event.rel_time[-1] - event.rel_time[0]),
                    "time_start": float(event.abs_time[0]),
                    "time_end": float(event.abs_time[-1]),
                    "voltage_start": float(event.voltage[0]),
                    "voltage_end": float(event.voltage[-1]),
                    "current_mean_a": float(np.mean(event.current)),
                    "temperature_mean_c": float(np.mean(event.temperature)),
                    "label_charge_count": int(event.label_charge_count),
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(EXTRACT_DIR / "charge_discharge_segment_metadata.csv", index=False)
    summary = (
        metadata.groupby(["battery", "event_type"], as_index=False)
        .agg(
            segments=("mat_step_index", "count"),
            rows=("sample_count", "sum"),
            min_rows=("sample_count", "min"),
            max_rows=("sample_count", "max"),
            median_rows=("sample_count", "median"),
            median_duration_s=("duration_s", "median"),
        )
        .sort_values(["battery", "event_type"])
    )
    summary.to_csv(EXTRACT_DIR / "extraction_summary.csv", index=False)
    return metadata


def build_weighted_patch_sequence(event: RandomWalkEvent, battery_time0: float) -> np.ndarray:
    raw_patch = build_patch_sequence(
        event.voltage.astype(np.float32),
        event.current.astype(np.float32),
        event.temperature.astype(np.float32),
        event.rel_time.astype(np.float32),
        event.abs_time.astype(np.float64),
        battery_time0,
    )

    v_scaled = minmax01(event.voltage)
    t_scaled = minmax01(event.temperature)
    c_scaled = maxabs01(event.current)
    r_scaled = minmax01(event.rel_time)
    norm_patch_rows: list[np.ndarray] = []
    token_count = math.ceil(event.sample_count / PATCH_SIZE)

    for patch_idx, start in enumerate(range(0, event.sample_count, PATCH_SIZE)):
        end = min(start + PATCH_SIZE, event.sample_count)
        pv = v_scaled[start:end]
        pt = t_scaled[start:end]
        pc = c_scaled[start:end]
        pr = r_scaled[start:end]
        norm_patch_rows.append(
            np.array(
                [
                    float(np.mean(pv)),
                    float(pv[-1] - pv[0]),
                    float(np.mean(pc)),
                    float(np.mean(np.abs(pc))),
                    float(np.mean(pt)),
                    float(pt[-1] - pt[0]),
                    float(pr[0]),
                    float(pr[-1]),
                    float((patch_idx + 1) / token_count),
                ],
                dtype=np.float32,
            )
        )
    norm_patch = np.stack(norm_patch_rows, axis=0)
    return np.concatenate([raw_patch, norm_patch], axis=1).astype(np.float32, copy=False)


def build_battery_sequences(
    battery: str,
    events: list[RandomWalkEvent],
) -> tuple[list[np.ndarray], pd.DataFrame]:
    checkpoint_counts, capacities, initial_capacity = load_reference_label_table(battery)
    battery_time0 = min(float(event.abs_time[0]) for event in events)

    sequences: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for event in events:
        assigned_checkpoint, target_capacity, target_soh = assign_next_checkpoint(
            event.label_charge_count,
            checkpoint_counts,
            capacities,
            initial_capacity,
        )
        seq = build_weighted_patch_sequence(event, battery_time0)
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


def concat_sequence_sets(parts: list[tuple[list[np.ndarray], pd.DataFrame]]) -> tuple[list[np.ndarray], pd.DataFrame]:
    sequences: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []
    for seq_part, meta_part in parts:
        sequences.extend(seq_part)
        metas.append(meta_part)
    return sequences, pd.concat(metas, ignore_index=True)


def add_block_weights(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    counts = (
        meta.groupby(["battery", "assigned_checkpoint_cycle"])
        .size()
        .rename("block_size")
        .reset_index()
    )
    meta = meta.merge(counts, on=["battery", "assigned_checkpoint_cycle"], how="left")
    meta["sample_weight"] = 1.0 / meta["block_size"].astype(np.float32)
    meta["sample_weight"] = meta["sample_weight"] / float(meta["sample_weight"].mean())
    return meta


class WeightedSequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], targets: np.ndarray, weights: np.ndarray, meta: pd.DataFrame):
        self.sequences = sequences
        self.targets = targets.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        return {
            "sequence": self.sequences[idx],
            "target": float(self.targets[idx]),
            "weight": float(self.weights[idx]),
            "battery": str(row["battery"]),
            "event_type": str(row["event_type"]),
            "event_order": int(row["event_order"]),
            "charge_cycle_for_label": int(row["charge_cycle_for_label"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": str(row["sample_id"]),
        }


def collate_batch(
    batch: list[dict[str, Any]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    list[int],
    list[int],
    list[int],
    list[str],
]:
    max_len = max(len(item["sequence"]) for item in batch)
    input_dim = batch[0]["sequence"].shape[1]
    batch_size = len(batch)
    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    y = torch.zeros(batch_size, dtype=torch.float32)
    w = torch.zeros(batch_size, dtype=torch.float32)
    batteries: list[str] = []
    event_types: list[str] = []
    event_orders: list[int] = []
    charge_cycles: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []
    for idx, item in enumerate(batch):
        seq = torch.from_numpy(item["sequence"]).float()
        seq_len = seq.shape[0]
        x[idx, :seq_len] = seq
        mask[idx, :seq_len] = True
        y[idx] = float(item["target"])
        w[idx] = float(item["weight"])
        batteries.append(item["battery"])
        event_types.append(item["event_type"])
        event_orders.append(int(item["event_order"]))
        charge_cycles.append(int(item["charge_cycle_for_label"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        sample_ids.append(item["sample_id"])
    return x, mask, y, w, batteries, event_types, event_orders, charge_cycles, checkpoints, sample_ids


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[int],
    list[int],
    list[int],
    list[str],
]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batteries: list[str] = []
    event_types: list[str] = []
    event_orders: list[int] = []
    charge_cycles: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []

    for batch in loader:
        batch_x, batch_mask, batch_y, batch_w, batch_battery, batch_type, batch_order, batch_cycle, batch_checkpoint, batch_sample_id = batch
        batch_x = batch_x.to(device)
        batch_mask = batch_mask.to(device)
        batch_y = batch_y.to(device)
        batch_w = batch_w.to(device)

        out = model(batch_x, batch_mask)
        per_sample = (out - batch_y) ** 2
        loss = (per_sample * batch_w).sum() / torch.clamp(batch_w.sum(), min=1.0e-8)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        preds.append(out.detach().cpu().numpy())
        targets.append(batch_y.detach().cpu().numpy())
        batteries.extend(batch_battery)
        event_types.extend(batch_type)
        event_orders.extend([int(x) for x in batch_order])
        charge_cycles.extend([int(x) for x in batch_cycle])
        checkpoints.extend([int(x) for x in batch_checkpoint])
        sample_ids.extend(batch_sample_id)

    return (
        total_loss / max(total_count, 1),
        np.concatenate(preds),
        np.concatenate(targets),
        batteries,
        event_types,
        event_orders,
        charge_cycles,
        checkpoints,
        sample_ids,
    )


def predictions_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batteries: list[str],
    event_types: list[str],
    event_orders: list[int],
    charge_cycles: list[int],
    checkpoints: list[int],
    sample_ids: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "battery": batteries,
            "event_type": event_types,
            "event_order": event_orders,
            "charge_cycle_for_label": charge_cycles,
            "assigned_checkpoint_cycle": checkpoints,
            "actual_soh": y_true,
            "predicted_soh": y_pred,
            "error": y_pred - y_true,
            "absolute_error": np.abs(y_pred - y_true),
        }
    )


def checkpoint_metrics(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    grouped = (
        pred_df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False)
        .agg(
            actual_soh=("actual_soh", "mean"),
            predicted_soh=("predicted_soh", "mean"),
            event_count=("sample_id", "count"),
            charge_events=("event_type", lambda x: int((x == "charge").sum())),
            discharge_events=("event_type", lambda x: int((x == "discharge").sum())),
        )
        .sort_values(["battery", "assigned_checkpoint_cycle"])
        .reset_index(drop=True)
    )
    grouped["error"] = grouped["predicted_soh"] - grouped["actual_soh"]
    grouped["absolute_error"] = grouped["error"].abs()
    return grouped, regression_metrics(grouped["actual_soh"].to_numpy(), grouped["predicted_soh"].to_numpy())


def save_loss_plot(history: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train", linewidth=2)
    axes[0].plot(history["epoch"], history["test_loss"], label="test", linewidth=2)
    axes[0].set_title("Weighted MSE loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_r2"], label="train", linewidth=2)
    axes[1].plot(history["epoch"], history["test_r2"], label="test", linewidth=2)
    axes[1].set_title("R2")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[2].plot(history["epoch"], history["train_mae"], label="train", linewidth=2)
    axes[2].plot(history["epoch"], history["test_mae"], label="test", linewidth=2)
    axes[2].set_title("MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(outdir / "training_dashboard.png", dpi=180)
    plt.close(fig)


def save_prediction_plots(pred_df: pd.DataFrame, checkpoint_df: pd.DataFrame, outdir: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = pred_df["event_type"].map({"charge": "#1f77b4", "discharge": "#d62728"}).fillna("#444444")
    axes[0, 0].scatter(pred_df["actual_soh"], pred_df["predicted_soh"], s=7, alpha=0.25, c=colors)
    lims = [
        min(pred_df["actual_soh"].min(), pred_df["predicted_soh"].min()),
        max(pred_df["actual_soh"].max(), pred_df["predicted_soh"].max()),
    ]
    axes[0, 0].plot(lims, lims, "k--", linewidth=1)
    axes[0, 0].set_title("Event-level actual vs predicted")
    axes[0, 0].set_xlabel("Actual SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)")
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].scatter(pred_df["charge_cycle_for_label"], pred_df["actual_soh"], s=6, alpha=0.18, label="actual", color="#111111")
    axes[0, 1].scatter(pred_df["charge_cycle_for_label"], pred_df["predicted_soh"], s=6, alpha=0.18, label="predicted", color="#2a9d8f")
    axes[0, 1].set_title("Event-level SOH over charge chronology")
    axes[0, 1].set_xlabel("Charge count used for label assignment")
    axes[0, 1].set_ylabel("SOH (%)")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["actual_soh"], marker="o", label="actual", linewidth=2)
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["predicted_soh"], marker="o", label="predicted", linewidth=2)
    axes[1, 0].set_title("Checkpoint-aggregated predictions")
    axes[1, 0].set_xlabel("Benchmark checkpoint charge count")
    axes[1, 0].set_ylabel("SOH (%)")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].hist(pred_df["error"], bins=60, color="#457b9d", alpha=0.85)
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Event-level residuals")
    axes[1, 1].set_xlabel("Prediction error (SOH points)")
    axes[1, 1].grid(True, alpha=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "prediction_report.png", dpi=180)
    plt.close(fig)


def save_comparison_plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(summary))
    labels = summary["run"].tolist()
    axes[0].bar(x, summary["test_cycle_r2"], color="#2a9d8f")
    axes[0].set_title("Test cycle R2")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, summary["test_cycle_mae"], color="#457b9d")
    axes[1].set_title("Test cycle MAE")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[2].bar(x, summary["test_checkpoint_r2"], color="#f4a261")
    axes[2].set_title("Checkpoint-aggregated R2")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "comparison_summary.png", dpi=180)
    plt.close(fig)


def run_split(name: str, split: dict[str, tuple[str, ...]], sequence_cache: dict[str, tuple[list[np.ndarray], pd.DataFrame]]) -> dict[str, Any]:
    outdir = OUTDIR / name
    outdir.mkdir(parents=True, exist_ok=True)

    train_sequences, train_meta = concat_sequence_sets([sequence_cache[b] for b in split["train"]])
    test_sequences, test_meta = concat_sequence_sets([sequence_cache[b] for b in split["test"]])
    train_meta = add_block_weights(train_meta)
    test_meta = add_block_weights(test_meta)

    mean, std = compute_standardization(train_sequences)
    apply_standardization(train_sequences, mean, std)
    apply_standardization(test_sequences, mean, std)

    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    train_ds = WeightedSequenceDataset(train_sequences, train_y_norm, train_w, train_meta)
    test_ds = WeightedSequenceDataset(test_sequences, test_y_norm, test_w, test_meta)
    train_loader = DataLoader(train_ds, batch_size=SPEC.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=SPEC.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cpu")
    input_dim = train_sequences[0].shape[1]
    model = EventTransformer(input_dim=input_dim, spec=SPEC).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC.learning_rate, weight_decay=SPEC.weight_decay)
    history_rows: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_test_r2 = -np.inf

    for epoch in range(1, SPEC.epochs + 1):
        train_loss, train_pred_norm, train_true_norm, *_ = run_epoch(model, train_loader, optimizer, device)
        test_loss, test_pred_norm, test_true_norm, *_ = run_epoch(model, test_loader, None, device)
        train_pred = destandardize(train_pred_norm, y_mean, y_std)
        train_true = destandardize(train_true_norm, y_mean, y_std)
        test_pred = destandardize(test_pred_norm, y_mean, y_std)
        test_true = destandardize(test_true_norm, y_mean, y_std)
        train_metrics = regression_metrics(train_true, train_pred)
        test_metrics = regression_metrics(test_true, test_pred)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
            }
        )
        if test_metrics["r2"] > best_test_r2:
            best_test_r2 = test_metrics["r2"]
            best_state = {
                "epoch": epoch,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        print(
            f"{name} epoch {epoch}/{SPEC.epochs}: "
            f"train_r2={train_metrics['r2']:.4f} test_r2={test_metrics['r2']:.4f} "
            f"test_mae={test_metrics['mae']:.3f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    train_eval = run_epoch(model, train_loader, None, device)
    test_eval = run_epoch(model, test_loader, None, device)
    _, train_pred_norm, train_true_norm, train_battery, train_type, train_order, train_cycle, train_checkpoint, train_ids = train_eval
    _, test_pred_norm, test_true_norm, test_battery, test_type, test_order, test_cycle, test_checkpoint, test_ids = test_eval
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)

    train_pred_df = predictions_frame(
        train_true, train_pred, train_battery, train_type, train_order, train_cycle, train_checkpoint, train_ids
    )
    test_pred_df = predictions_frame(
        test_true, test_pred, test_battery, test_type, test_order, test_cycle, test_checkpoint, test_ids
    )
    checkpoint_df, checkpoint_metric = checkpoint_metrics(test_pred_df)
    train_metric = regression_metrics(train_true, train_pred)
    test_metric = regression_metrics(test_true, test_pred)

    history = pd.DataFrame(history_rows)
    history.to_csv(outdir / "training_history.csv", index=False)
    train_pred_df.to_csv(outdir / "train_predictions.csv", index=False)
    test_pred_df.to_csv(outdir / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(outdir / "test_checkpoint_predictions.csv", index=False)

    dataset_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "battery_set": "+".join(split["train"]),
                "sample_count": len(train_meta),
                "charge_samples": int((train_meta["event_type"] == "charge").sum()),
                "discharge_samples": int((train_meta["event_type"] == "discharge").sum()),
                "unique_checkpoint_labels": int(train_meta[["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "median_token_count": float(train_meta["token_count"].median()),
                "max_token_count": int(train_meta["token_count"].max()),
                "median_block_size": float(train_meta["block_size"].median()),
                "max_block_size": int(train_meta["block_size"].max()),
            },
            {
                "split": "test",
                "battery_set": "+".join(split["test"]),
                "sample_count": len(test_meta),
                "charge_samples": int((test_meta["event_type"] == "charge").sum()),
                "discharge_samples": int((test_meta["event_type"] == "discharge").sum()),
                "unique_checkpoint_labels": int(test_meta[["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "median_token_count": float(test_meta["token_count"].median()),
                "max_token_count": int(test_meta["token_count"].max()),
                "median_block_size": float(test_meta["block_size"].median()),
                "max_block_size": int(test_meta["block_size"].max()),
            },
        ]
    )
    dataset_summary.to_csv(outdir / "dataset_summary.csv", index=False)

    metrics = {
        "run": name,
        "split": split,
        "spec": asdict(SPEC),
        "input_dim": input_dim,
        "parameter_count": parameter_count(model),
        "best_epoch_by_test_r2": int(best_state["epoch"]) if best_state is not None else SPEC.epochs,
        "train_metrics_event_level": train_metric,
        "test_metrics_event_level": test_metric,
        "test_metrics_checkpoint_aggregated": checkpoint_metric,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, outdir / "model.pt")

    save_loss_plot(history, outdir)
    save_prediction_plots(test_pred_df, checkpoint_df, outdir, title=name)

    return {
        "run": name,
        "train_samples": len(train_meta),
        "test_samples": len(test_meta),
        "train_charge_samples": int((train_meta["event_type"] == "charge").sum()),
        "train_discharge_samples": int((train_meta["event_type"] == "discharge").sum()),
        "test_charge_samples": int((test_meta["event_type"] == "charge").sum()),
        "test_discharge_samples": int((test_meta["event_type"] == "discharge").sum()),
        "test_unique_checkpoints": int(test_meta[["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
        "best_epoch_by_test_r2": metrics["best_epoch_by_test_r2"],
        "test_cycle_r2": test_metric["r2"],
        "test_cycle_mae": test_metric["mae"],
        "test_cycle_rmse": test_metric["rmse"],
        "test_checkpoint_r2": checkpoint_metric["r2"],
        "test_checkpoint_mae": checkpoint_metric["mae"],
        "test_checkpoint_rmse": checkpoint_metric["rmse"],
        "output_dir": str(outdir),
    }


def main() -> None:
    set_seed()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_events = {battery: extract_events_from_mat(battery) for battery in BATTERIES}
    export_extracted_segments(all_events)
    sequence_cache = {battery: build_battery_sequences(battery, all_events[battery]) for battery in BATTERIES}
    summaries = []
    for split_name, split in SPLITS.items():
        summaries.append(run_split(split_name, split, sequence_cache))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTDIR / "comparison_summary.csv", index=False)
    save_comparison_plot(summary)
    (OUTDIR / "run_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
