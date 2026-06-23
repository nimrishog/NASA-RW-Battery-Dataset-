from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_full_event_token_transformers import (
    EventTransformer,
    apply_standardization,
    build_patch_sequence,
    compute_standardization,
    concat_sequence_sets,
    destandardize,
    next_benchmark_label_map,
    parameter_count,
    regression_metrics,
    standardize_target,
)


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
PKL_DIR = BASE / "charge pkl with temperature"
OUTDIR = BASE / "cycle_level_weighted_transformer"
TRAIN_BATTERIES = ("RW9", "RW10", "RW11")
TEST_BATTERIES = ("RW12",)


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
    name="cycle_patch30_weighted",
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def build_weighted_patch_sequence(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    rel_time: np.ndarray,
    abs_time: np.ndarray,
    battery_time0: float,
) -> np.ndarray:
    raw_patch = build_patch_sequence(voltage, current, temperature, rel_time, abs_time, battery_time0)

    v_scaled = minmax01(voltage)
    t_scaled = minmax01(temperature)
    c_scaled = maxabs01(current)
    r_scaled = minmax01(rel_time)

    norm_patch_rows: list[np.ndarray] = []
    for patch_idx, start in enumerate(range(0, len(voltage), 30)):
        end = min(start + 30, len(voltage))
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
                    float((patch_idx + 1) / math.ceil(len(voltage) / 30)),
                ],
                dtype=np.float32,
            )
        )
    norm_patch = np.stack(norm_patch_rows, axis=0)
    return np.concatenate([raw_patch, norm_patch], axis=1).astype(np.float32, copy=False)


def build_battery_sequences(battery: str) -> tuple[list[np.ndarray], pd.DataFrame]:
    df = (
        pd.read_pickle(PKL_DIR / f"{battery}_charge_temp_datacapa.pkl")
        .sort_values(["cycle", "relTime", "time"])
        .reset_index(drop=True)
    )
    cycles = df["cycle"].to_numpy(dtype=np.int32)
    voltage = df["Voltage"].to_numpy(dtype=np.float32)
    current = df["Current"].to_numpy(dtype=np.float32)
    temperature = df["Temperature"].to_numpy(dtype=np.float32)
    rel_time = df["relTime"].to_numpy(dtype=np.float32)
    abs_time = df["time"].to_numpy(dtype=np.float64)

    unique_cycles, starts = np.unique(cycles, return_index=True)
    ends = np.r_[starts[1:], len(df)]
    cycle_to_soh, cycle_to_capacity, cycle_to_checkpoint, initial_capacity = next_benchmark_label_map(
        battery, int(unique_cycles.max())
    )
    battery_time0 = float(abs_time.min())

    sequences: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for cycle, start, end in zip(unique_cycles.tolist(), starts.tolist(), ends.tolist()):
        v = voltage[start:end]
        c = current[start:end]
        t = temperature[start:end]
        rt = rel_time[start:end]
        at = abs_time[start:end]
        seq = build_weighted_patch_sequence(v, c, t, rt, at, battery_time0)
        sequences.append(seq)
        meta_rows.append(
            {
                "battery": battery,
                "charge_cycle": int(cycle),
                "assigned_checkpoint_cycle": int(cycle_to_checkpoint[int(cycle)]),
                "sample_id": f"{battery}_cycle_{int(cycle)}",
                "target_capacity_ah": float(cycle_to_capacity[int(cycle)]),
                "target_soh_percent": float(cycle_to_soh[int(cycle)]),
                "token_count": int(len(seq)),
                "raw_row_count": int(end - start),
                "initial_capacity_ah": float(initial_capacity),
            }
        )
    return sequences, pd.DataFrame(meta_rows)


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
            "charge_cycle": int(row["charge_cycle"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": str(row["sample_id"]),
        }


def collate_batch(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int], list[int], list[str]]:
    max_len = max(len(item["sequence"]) for item in batch)
    input_dim = batch[0]["sequence"].shape[1]
    batch_size = len(batch)
    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    y = torch.zeros(batch_size, dtype=torch.float32)
    w = torch.zeros(batch_size, dtype=torch.float32)
    batteries: list[str] = []
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
        charge_cycles.append(int(item["charge_cycle"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        sample_ids.append(item["sample_id"])
    return x, mask, y, w, batteries, charge_cycles, checkpoints, sample_ids


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[int], list[int], list[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batteries: list[str] = []
    charge_cycles: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []

    with torch.set_grad_enabled(training):
        for batch_x, batch_mask, batch_y, batch_w, batch_battery, batch_cycle, batch_checkpoint, batch_id in loader:
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
            charge_cycles.extend([int(x) for x in batch_cycle])
            checkpoints.extend([int(x) for x in batch_checkpoint])
            sample_ids.extend(batch_id)

    return total_loss / max(total_count, 1), np.concatenate(preds), np.concatenate(targets), batteries, charge_cycles, checkpoints, sample_ids


def build_prediction_frame(
    batteries: list[str],
    charge_cycles: list[int],
    checkpoints: list[int],
    sample_ids: list[str],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery": batteries,
            "charge_cycle": charge_cycles,
            "assigned_checkpoint_cycle": checkpoints,
            "sample_id": sample_ids,
            "actual_soh_percent": actual,
            "predicted_soh_percent": predicted,
            "absolute_error_pp": np.abs(predicted - actual),
        }
    )


def aggregate_checkpoint_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pred_df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False)
        .agg(
            actual_soh_percent=("actual_soh_percent", "mean"),
            predicted_soh_percent=("predicted_soh_percent", "mean"),
            cycle_count=("sample_id", "count"),
        )
        .sort_values(["battery", "assigned_checkpoint_cycle"])
        .reset_index(drop=True)
    )


def save_loss_plot(history: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train_weighted_loss", linewidth=2)
    ax.plot(history["epoch"], history["test_loss"], label="test_weighted_loss", linewidth=2)
    ax.set_title("Cycle-level weighted transformer loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE on standardized SOH")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "loss_curves.png", dpi=180)
    plt.close(fig)


def save_prediction_plots(test_cycle_df: pd.DataFrame, test_checkpoint_df: pd.DataFrame, outdir: Path) -> None:
    y_true = test_cycle_df["actual_soh_percent"].to_numpy(dtype=np.float64)
    y_pred = test_cycle_df["predicted_soh_percent"].to_numpy(dtype=np.float64)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sample = np.linspace(0, len(y_true) - 1, min(len(y_true), 5000)).astype(int)
    axes[0].scatter(y_true[sample], y_pred[sample], s=14, alpha=0.30)
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[0].set_title("RW12 cycle-level predicted vs actual SOH")
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")
    residuals = y_pred[sample] - y_true[sample]
    axes[1].scatter(y_true[sample], residuals, s=14, alpha=0.30)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("RW12 cycle-level residuals")
    axes[1].set_xlabel("Actual SOH (%)")
    axes[1].set_ylabel("Error (pp)")
    fig.savefig(outdir / "cycle_predictions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_df = test_checkpoint_df.sort_values("assigned_checkpoint_cycle").reset_index(drop=True)
    ax.plot(plot_df["assigned_checkpoint_cycle"], plot_df["actual_soh_percent"], marker="o", linewidth=1.8, label="actual")
    ax.plot(plot_df["assigned_checkpoint_cycle"], plot_df["predicted_soh_percent"], marker="o", linewidth=1.8, label="predicted")
    ax.set_title("RW12 checkpoint-aggregated SOH")
    ax.set_xlabel("Assigned benchmark checkpoint cycle")
    ax.set_ylabel("SOH (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "checkpoint_predictions.png", dpi=180)
    plt.close(fig)


def save_flow_diagram(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 7), constrained_layout=True)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    blocks = [
        (0.5, 6.8, 2.2, 1.2, "Raw NASA\nRW9-RW12 .mat"),
        (3.0, 6.8, 2.3, 1.2, "Extract charge\n(random walk)\nsteps"),
        (5.7, 6.8, 2.5, 1.2, "One charge cycle\n= one sample\n(2..301 rows)"),
        (8.7, 6.8, 2.6, 1.2, "Next benchmark\nSOH label\nfrom reference\ndischarge"),
        (11.8, 6.8, 2.7, 1.2, "Inverse block\nweight = 1 / cycles\nsharing label"),
        (15.0, 6.8, 2.2, 1.2, "Train/test split\nTrain: RW9-11\nTest: RW12"),
        (0.9, 3.8, 2.6, 1.5, "Patch each cycle\ninto 30-row tokens"),
        (4.0, 3.8, 2.8, 1.5, "Token features:\nraw + per-cycle\nnormalized patch\nfeatures"),
        (7.5, 3.8, 2.8, 1.5, "[CLS] + positional\nencoding + padding\nmask"),
        (11.0, 3.4, 3.2, 2.3, "Transformer encoder\nrepeated 3x\nMulti-head self-attention\n+ FFN + residuals"),
        (15.0, 3.8, 2.2, 1.5, "SOH prediction\nfor the cycle"),
        (11.0, 0.8, 3.2, 1.5, "Weighted MSE loss\n+ checkpoint\naggregation"),
    ]

    for x, y, w, h, label in blocks:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.5, edgecolor="#1d3557", facecolor="#edf6f9")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.7, 7.4), (3.0, 7.4)),
        ((5.3, 7.4), (5.7, 7.4)),
        ((8.2, 7.4), (8.7, 7.4)),
        ((11.3, 7.4), (11.8, 7.4)),
        ((14.5, 7.4), (15.0, 7.4)),
        ((16.1, 6.8), (16.1, 5.4)),
        ((16.1, 4.6), (15.0, 4.6)),
        ((14.2, 4.6), (10.3, 4.6)),
        ((7.5, 4.6), (6.8, 4.6)),
        ((4.0, 4.6), (3.5, 4.6)),
        ((2.2, 6.8), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 5.3)),
        ((2.2, 5.3), (2.2, 4.6)),
        ((14.1, 3.4), (14.1, 2.3)),
        ((15.0, 1.55), (15.0, 1.55)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#457b9d"))

    ax.add_patch(FancyArrowPatch((2.2, 6.8), (2.2, 5.3), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#457b9d"))
    ax.add_patch(FancyArrowPatch((2.2, 5.3), (2.2, 5.25), arrowstyle="-", linewidth=0))
    ax.add_patch(FancyArrowPatch((2.2, 5.25), (2.2, 5.2), arrowstyle="-", linewidth=0))
    ax.add_patch(FancyArrowPatch((2.2, 5.2), (2.2, 5.15), arrowstyle="-", linewidth=0))
    ax.add_patch(FancyArrowPatch((2.2, 5.15), (2.2, 5.1), arrowstyle="-", linewidth=0))
    ax.add_patch(FancyArrowPatch((2.2, 5.1), (2.2, 5.05), arrowstyle="-", linewidth=0))
    ax.add_patch(FancyArrowPatch((2.2, 5.05), (2.2, 5.0), arrowstyle="-", linewidth=0))
    ax.add_patch(FancyArrowPatch((2.2, 5.0), (2.2, 4.55), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#457b9d"))
    ax.add_patch(FancyArrowPatch((16.1, 6.8), (16.1, 5.3), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#457b9d"))
    ax.add_patch(FancyArrowPatch((16.1, 5.3), (16.1, 4.55), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#457b9d"))
    ax.add_patch(FancyArrowPatch((14.1, 3.4), (14.1, 2.35), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#457b9d"))

    encoder = FancyBboxPatch((11.35, 3.75), 2.5, 1.55, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2, edgecolor="#6d597a", facecolor="#fff3b0")
    ax.add_patch(encoder)
    ax.text(12.6, 4.52, "Encoder block x 3", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(12.6, 4.08, "Self-attention\nResidual + Norm\nFFN", ha="center", va="center", fontsize=9)

    fig.savefig(outdir / "process_flow.png", dpi=180)
    plt.close(fig)

    mermaid = """```mermaid
flowchart LR
    A[Raw NASA RW9-RW12 .mat] --> B[Extract charge random-walk steps]
    B --> C[One charge cycle = one sample]
    C --> D[Assign next benchmark SOH label]
    D --> E[Compute inverse block weight]
    E --> F[Split by battery: Train RW9-11, Test RW12]
    C --> G[Patch cycle into 30-row tokens]
    G --> H[Build raw + per-cycle normalized token features]
    H --> I[Add CLS token, positional encoding, padding mask]
    I --> J[Transformer encoder x3]
    J --> K[Cycle-level SOH prediction]
    K --> L[Weighted MSE loss]
    K --> M[Aggregate cycle predictions by benchmark checkpoint]
```"""
    (outdir / "process_flow.md").write_text(mermaid, encoding="utf-8")


def train_and_evaluate() -> dict[str, Any]:
    train_parts = [build_battery_sequences(battery) for battery in TRAIN_BATTERIES]
    test_parts = [build_battery_sequences(battery) for battery in TEST_BATTERIES]
    train_sequences, train_meta = concat_sequence_sets(train_parts)
    test_sequences, test_meta = concat_sequence_sets(test_parts)
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

    history_rows: list[dict[str, float | int]] = []
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
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
            }
        )
        print(
            f"epoch {epoch}/{SPEC.epochs}: train_r2={train_metrics['r2']:.4f} "
            f"test_r2={test_metrics['r2']:.4f} train_mae={train_metrics['mae']:.4f} "
            f"test_mae={test_metrics['mae']:.4f}",
            flush=True,
        )

    _, train_pred_norm, train_true_norm, train_batteries, train_cycles, train_checkpoints, train_ids = run_epoch(model, train_loader, None, device)
    _, test_pred_norm, test_true_norm, test_batteries, test_cycles, test_checkpoints, test_ids = run_epoch(model, test_loader, None, device)
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)

    train_cycle_df = build_prediction_frame(train_batteries, train_cycles, train_checkpoints, train_ids, train_true, train_pred)
    test_cycle_df = build_prediction_frame(test_batteries, test_cycles, test_checkpoints, test_ids, test_true, test_pred)
    test_checkpoint_df = aggregate_checkpoint_predictions(test_cycle_df)

    return {
        "state_dict": model.state_dict(),
        "history": pd.DataFrame(history_rows),
        "train_meta": train_meta,
        "test_meta": test_meta,
        "train_cycle_df": train_cycle_df,
        "test_cycle_df": test_cycle_df,
        "test_checkpoint_df": test_checkpoint_df,
        "train_metrics": regression_metrics(train_true, train_pred),
        "test_metrics": regression_metrics(test_true, test_pred),
        "test_checkpoint_metrics": regression_metrics(
            test_checkpoint_df["actual_soh_percent"].to_numpy(dtype=np.float64),
            test_checkpoint_df["predicted_soh_percent"].to_numpy(dtype=np.float64),
        ),
        "parameter_count": parameter_count(model),
        "input_dim": input_dim,
        "target_mean": y_mean,
        "target_std": y_std,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    result = train_and_evaluate()
    result["history"].to_csv(OUTDIR / "training_history.csv", index=False)
    result["train_meta"].to_csv(OUTDIR / "train_metadata.csv", index=False)
    result["test_meta"].to_csv(OUTDIR / "test_metadata.csv", index=False)
    result["train_cycle_df"].to_csv(OUTDIR / "train_cycle_predictions.csv", index=False)
    result["test_cycle_df"].to_csv(OUTDIR / "test_cycle_predictions.csv", index=False)
    result["test_checkpoint_df"].to_csv(OUTDIR / "test_checkpoint_predictions.csv", index=False)
    torch.save(result["state_dict"], OUTDIR / "model.pt")

    save_loss_plot(result["history"], OUTDIR)
    save_prediction_plots(result["test_cycle_df"], result["test_checkpoint_df"], OUTDIR)
    save_flow_diagram(OUTDIR)

    dataset_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "sample_count": len(result["train_meta"]),
                "unique_batteries": int(result["train_meta"]["battery"].nunique()),
                "unique_checkpoint_labels": int(result["train_meta"][["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "median_token_count": float(result["train_meta"]["token_count"].median()),
                "max_token_count": int(result["train_meta"]["token_count"].max()),
                "median_block_size": float(result["train_meta"]["block_size"].median()),
                "max_block_size": int(result["train_meta"]["block_size"].max()),
            },
            {
                "split": "test",
                "sample_count": len(result["test_meta"]),
                "unique_batteries": int(result["test_meta"]["battery"].nunique()),
                "unique_checkpoint_labels": int(result["test_meta"][["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "median_token_count": float(result["test_meta"]["token_count"].median()),
                "max_token_count": int(result["test_meta"]["token_count"].max()),
                "median_block_size": float(result["test_meta"]["block_size"].median()),
                "max_block_size": int(result["test_meta"]["block_size"].max()),
            },
        ]
    )
    dataset_summary.to_csv(OUTDIR / "dataset_summary.csv", index=False)

    metrics = {
        "spec": asdict(SPEC),
        "input_dim": result["input_dim"],
        "parameter_count": result["parameter_count"],
        "train_metrics_cycle_level": result["train_metrics"],
        "test_metrics_cycle_level": result["test_metrics"],
        "test_metrics_checkpoint_aggregated": result["test_checkpoint_metrics"],
    }
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
