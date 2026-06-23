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
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
PKL_DIR = BASE / "charge pkl with temperature"
ANALYSIS_DIR = BASE / "Analysis"
OUTDIR = BASE / "full_event_token_transformers"

TRAIN_BATTERIES = ("RW9", "RW10", "RW11")
TEST_BATTERIES = ("RW12",)
PATCH_SIZE = 30


@dataclass(frozen=True)
class DenseSpec:
    name: str
    token_mode: str
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


SPECS = [
    DenseSpec("patch30_deep", "patch30", 96, 8, 4, 192, 0.10, 3.0e-4, 1.0e-4, 18, 192, 11),
    DenseSpec("row_tokens_deep", "row", 64, 8, 3, 160, 0.10, 3.0e-4, 1.0e-4, 8, 96, 301),
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    pct_err = abs_err / np.clip(np.abs(y_true), 1.0e-8, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(pct_err.mean() * 100.0),
        "within_2pct": float((pct_err <= 0.02).mean()),
        "within_5pct": float((pct_err <= 0.05).mean()),
        "within_10pct": float((pct_err <= 0.10).mean()),
    }


def create_sinusoidal_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class EventTransformer(nn.Module):
    def __init__(self, input_dim: int, spec: DenseSpec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.input_proj = nn.Linear(input_dim, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        pe = create_sinusoidal_positional_encoding(spec.max_tokens + 1, spec.d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.nhead,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=spec.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(spec.d_model),
            nn.Linear(spec.d_model, spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_model, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.input_norm(self.input_proj(x))
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones((mask.size(0), 1), device=mask.device, dtype=torch.bool)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        tokens = tokens + self.pe[:, : tokens.size(1)]
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return self.head(encoded[:, 0]).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def next_benchmark_label_map(battery: str, max_cycle: int) -> tuple[dict[int, float], dict[int, float], dict[int, int], float]:
    ref = pd.read_csv(ANALYSIS_DIR / battery / "reference_capacity_summary.csv")
    grouped = (
        ref.groupby("charge_cycle_count_before_reference", as_index=False)
        .agg(capacity_ah=("capacity_ah", "mean"))
        .sort_values("charge_cycle_count_before_reference")
        .reset_index(drop=True)
    )
    initial_capacity = float(grouped.loc[grouped["charge_cycle_count_before_reference"].eq(0), "capacity_ah"].mean())
    grouped = grouped[grouped["charge_cycle_count_before_reference"] > 0].reset_index(drop=True)
    checkpoint_cycles = grouped["charge_cycle_count_before_reference"].to_numpy(dtype=np.int64)
    capacities = grouped["capacity_ah"].to_numpy(dtype=np.float64)
    soh_values = capacities / initial_capacity * 100.0

    cycle_to_soh: dict[int, float] = {}
    cycle_to_capacity: dict[int, float] = {}
    cycle_to_checkpoint: dict[int, int] = {}
    for cycle in range(1, max_cycle + 1):
        idx = int(np.searchsorted(checkpoint_cycles, cycle, side="left"))
        if idx >= len(checkpoint_cycles):
            idx = len(checkpoint_cycles) - 1
        cycle_to_soh[cycle] = float(soh_values[idx])
        cycle_to_capacity[cycle] = float(capacities[idx])
        cycle_to_checkpoint[cycle] = int(checkpoint_cycles[idx])
    return cycle_to_soh, cycle_to_capacity, cycle_to_checkpoint, initial_capacity


def diff_with_zero(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.size > 1:
        out[1:] = np.diff(values).astype(np.float32)
    return out


def build_row_sequence(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    rel_time: np.ndarray,
    abs_time: np.ndarray,
    initial_capacity: float,
    battery_time0: float,
) -> np.ndarray:
    dt = np.clip(diff_with_zero(rel_time), 0.0, None)
    cumulative_charge = np.cumsum((-current * dt) / 3600.0, dtype=np.float32)
    cumulative_energy = np.cumsum((voltage * (-current) * dt) / 3600.0, dtype=np.float32)
    delta_v = diff_with_zero(voltage)
    delta_temp = diff_with_zero(temperature)
    elapsed_days = ((abs_time - battery_time0) / 86400.0).astype(np.float32)
    voltage_position = np.clip((voltage - 3.2) / 1.0, 0.0, 1.0).astype(np.float32)
    current_rate = (current / max(initial_capacity, 1.0e-8)).astype(np.float32)
    return np.column_stack(
        [
            voltage.astype(np.float32, copy=False),
            current.astype(np.float32, copy=False),
            temperature.astype(np.float32, copy=False),
            rel_time.astype(np.float32, copy=False),
            dt,
            cumulative_charge,
            cumulative_energy,
            delta_v,
            delta_temp,
            elapsed_days,
            voltage_position,
            current_rate,
        ]
    ).astype(np.float32, copy=False)


def build_patch_sequence(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    rel_time: np.ndarray,
    abs_time: np.ndarray,
    battery_time0: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(voltage), PATCH_SIZE):
        end = min(start + PATCH_SIZE, len(voltage))
        v = voltage[start:end]
        c = current[start:end]
        t = temperature[start:end]
        rt = rel_time[start:end]
        at = abs_time[start:end]
        dt = np.diff(rt).astype(np.float32)
        dt_clipped = np.clip(dt, 0.0, None)
        charge_ah = float(np.sum((-c[:-1]) * dt_clipped) / 3600.0) if len(v) > 1 else 0.0
        energy_wh = float(np.sum((v[:-1] * (-c[:-1])) * dt_clipped) / 3600.0) if len(v) > 1 else 0.0
        duration_s = float(rt[-1] - rt[0]) if len(v) > 1 else 0.0
        dv_dt = float((v[-1] - v[0]) / max(duration_s, 1.0e-8))
        dtemp_dt = float((t[-1] - t[0]) / max(duration_s, 1.0e-8))
        elapsed_start = float((at[0] - battery_time0) / 86400.0)
        elapsed_end = float((at[-1] - battery_time0) / 86400.0)
        rows.append(
            np.array(
                [
                    float(len(v)),
                    duration_s,
                    charge_ah,
                    energy_wh,
                    float(v[0]),
                    float(v[-1]),
                    float(np.mean(v)),
                    float(v[-1] - v[0]),
                    float(np.mean(c)),
                    float(np.std(c)),
                    float(np.mean(np.abs(c))),
                    float(np.mean(t)),
                    float(t[-1] - t[0]),
                    dv_dt,
                    dtemp_dt,
                    elapsed_start,
                    elapsed_end,
                ],
                dtype=np.float32,
            )
        )
    return np.stack(rows, axis=0)


def build_battery_sequences(battery: str, token_mode: str) -> tuple[list[np.ndarray], pd.DataFrame]:
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
        if token_mode == "row":
            seq = build_row_sequence(v, c, t, rt, at, initial_capacity, battery_time0)
        elif token_mode == "patch30":
            seq = build_patch_sequence(v, c, t, rt, at, battery_time0)
        else:
            raise ValueError(f"Unsupported token mode: {token_mode}")
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


def compute_standardization(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = sequences[0].shape[1]
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for seq in sequences:
        clean = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        total += clean.sum(axis=0)
        total_sq += np.square(clean).sum(axis=0)
        count += len(clean)
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1.0e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardization(sequences: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> None:
    for idx, seq in enumerate(sequences):
        clean = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        sequences[idx] = ((clean - mean) / std).astype(np.float32, copy=False)


def standardize_target(train_y: np.ndarray, other_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((other_y - mean) / std).astype(np.float32), mean, std


class VariableSequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], targets: np.ndarray, meta: pd.DataFrame):
        self.sequences = sequences
        self.targets = targets.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        return {
            "sequence": self.sequences[idx],
            "target": float(self.targets[idx]),
            "battery": str(row["battery"]),
            "charge_cycle": int(row["charge_cycle"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": str(row["sample_id"]),
        }


def collate_batch(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int], list[int], list[str]]:
    max_len = max(len(item["sequence"]) for item in batch)
    input_dim = batch[0]["sequence"].shape[1]
    batch_size = len(batch)
    x = torch.zeros((batch_size, max_len, input_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    y = torch.zeros(batch_size, dtype=torch.float32)
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
        batteries.append(item["battery"])
        charge_cycles.append(int(item["charge_cycle"]))
        checkpoints.append(int(item["assigned_checkpoint_cycle"]))
        sample_ids.append(item["sample_id"])
    return x, mask, y, batteries, charge_cycles, checkpoints, sample_ids


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
        for batch_x, batch_mask, batch_y, batch_battery, batch_cycle, batch_checkpoint, batch_id in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x, batch_mask)
            loss = nn.functional.mse_loss(out, batch_y)
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


def destandardize(y_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_norm * std + mean


def aggregate_checkpoint_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pred_df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False)
        .agg(
            actual_soh_percent=("actual_soh_percent", "mean"),
            predicted_soh_percent=("predicted_soh_percent", "mean"),
            event_count=("sample_id", "count"),
        )
        .sort_values(["battery", "assigned_checkpoint_cycle"])
        .reset_index(drop=True)
    )


def save_loss_plot(history: pd.DataFrame, outdir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train_loss", linewidth=2)
    ax.plot(history["epoch"], history["test_loss"], label="test_loss", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on standardized SOH")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "loss_curves.png", dpi=180)
    plt.close(fig)


def save_event_prediction_plot(pred_df: pd.DataFrame, outdir: Path, title_prefix: str) -> None:
    y_true = pred_df["actual_soh_percent"].to_numpy(dtype=np.float64)
    y_pred = pred_df["predicted_soh_percent"].to_numpy(dtype=np.float64)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(y_true, y_pred, s=10, alpha=0.22)
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[0].set_title(f"{title_prefix}: event predictions")
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")

    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, s=10, alpha=0.22)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title(f"{title_prefix}: event residuals")
    axes[1].set_xlabel("Actual SOH (%)")
    axes[1].set_ylabel("Error (pp)")
    fig.savefig(outdir / "event_predictions.png", dpi=180)
    plt.close(fig)


def save_checkpoint_plot(pred_df: pd.DataFrame, outdir: Path, title_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_df = pred_df.sort_values(["battery", "assigned_checkpoint_cycle"]).reset_index(drop=True)
    ax.plot(plot_df["assigned_checkpoint_cycle"], plot_df["actual_soh_percent"], marker="o", linewidth=1.8, label="actual")
    ax.plot(plot_df["assigned_checkpoint_cycle"], plot_df["predicted_soh_percent"], marker="o", linewidth=1.8, label="predicted")
    ax.set_title(f"{title_prefix}: checkpoint-aggregated RW12 SOH")
    ax.set_xlabel("Assigned benchmark checkpoint cycle")
    ax.set_ylabel("SOH (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "checkpoint_predictions.png", dpi=180)
    plt.close(fig)


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


def save_metadata_summary(train_meta: pd.DataFrame, test_meta: pd.DataFrame, outdir: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "sample_count": len(train_meta),
                "unique_batteries": int(train_meta["battery"].nunique()),
                "unique_checkpoint_labels": int(train_meta[["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "min_token_count": int(train_meta["token_count"].min()),
                "median_token_count": float(train_meta["token_count"].median()),
                "max_token_count": int(train_meta["token_count"].max()),
            },
            {
                "split": "test",
                "sample_count": len(test_meta),
                "unique_batteries": int(test_meta["battery"].nunique()),
                "unique_checkpoint_labels": int(test_meta[["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "min_token_count": int(test_meta["token_count"].min()),
                "median_token_count": float(test_meta["token_count"].median()),
                "max_token_count": int(test_meta["token_count"].max()),
            },
        ]
    )
    summary.to_csv(outdir / "dataset_summary.csv", index=False)


def train_and_evaluate(spec: DenseSpec) -> dict[str, Any]:
    train_parts = [build_battery_sequences(battery, spec.token_mode) for battery in TRAIN_BATTERIES]
    test_parts = [build_battery_sequences(battery, spec.token_mode) for battery in TEST_BATTERIES]
    train_sequences, train_meta = concat_sequence_sets(train_parts)
    test_sequences, test_meta = concat_sequence_sets(test_parts)

    mean, std = compute_standardization(train_sequences)
    apply_standardization(train_sequences, mean, std)
    apply_standardization(test_sequences, mean, std)

    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    train_ds = VariableSequenceDataset(train_sequences, train_y_norm, train_meta)
    test_ds = VariableSequenceDataset(test_sequences, test_y_norm, test_meta)
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    device = torch.device("cpu")
    input_dim = train_sequences[0].shape[1]
    model = EventTransformer(input_dim=input_dim, spec=spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)

    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, spec.epochs + 1):
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
            f"{spec.name} epoch {epoch}/{spec.epochs}: "
            f"train_r2={train_metrics['r2']:.4f} test_r2={test_metrics['r2']:.4f} "
            f"train_mae={train_metrics['mae']:.4f} test_mae={test_metrics['mae']:.4f}",
            flush=True,
        )

    _, train_pred_norm, train_true_norm, train_batteries, train_cycles, train_checkpoints, train_ids = run_epoch(model, train_loader, None, device)
    _, test_pred_norm, test_true_norm, test_batteries, test_cycles, test_checkpoints, test_ids = run_epoch(model, test_loader, None, device)
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)

    train_pred_df = build_prediction_frame(train_batteries, train_cycles, train_checkpoints, train_ids, train_true, train_pred)
    test_pred_df = build_prediction_frame(test_batteries, test_cycles, test_checkpoints, test_ids, test_true, test_pred)
    test_checkpoint_df = aggregate_checkpoint_predictions(test_pred_df)

    return {
        "spec": spec,
        "state_dict": model.state_dict(),
        "parameter_count": parameter_count(model),
        "history": pd.DataFrame(history_rows),
        "train_predictions": train_pred_df,
        "test_predictions": test_pred_df,
        "test_checkpoint_predictions": test_checkpoint_df,
        "train_metrics_event": regression_metrics(train_true, train_pred),
        "test_metrics_event": regression_metrics(test_true, test_pred),
        "test_metrics_checkpoint": regression_metrics(
            test_checkpoint_df["actual_soh_percent"].to_numpy(dtype=np.float64),
            test_checkpoint_df["predicted_soh_percent"].to_numpy(dtype=np.float64),
        ),
        "train_meta": train_meta,
        "test_meta": test_meta,
    }


def save_run_artifacts(result: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    spec: DenseSpec = result["spec"]
    result["history"].to_csv(outdir / "training_history.csv", index=False)
    result["train_predictions"].to_csv(outdir / "train_event_predictions.csv", index=False)
    result["test_predictions"].to_csv(outdir / "test_event_predictions.csv", index=False)
    result["test_checkpoint_predictions"].to_csv(outdir / "test_checkpoint_predictions.csv", index=False)
    result["train_meta"].to_csv(outdir / "train_metadata.csv", index=False)
    result["test_meta"].to_csv(outdir / "test_metadata.csv", index=False)
    save_loss_plot(result["history"], outdir, f"{spec.name}: full-event train/test loss")
    save_event_prediction_plot(result["test_predictions"], outdir, spec.name)
    save_checkpoint_plot(result["test_checkpoint_predictions"], outdir, spec.name)
    save_metadata_summary(result["train_meta"], result["test_meta"], outdir)
    torch.save(result["state_dict"], outdir / "model.pt")
    summary = {
        "spec": asdict(spec),
        "parameter_count": result["parameter_count"],
        "train_metrics_event": result["train_metrics_event"],
        "test_metrics_event": result["test_metrics_event"],
        "test_metrics_checkpoint": result["test_metrics_checkpoint"],
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_comparison(results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        spec: DenseSpec = result["spec"]
        rows.append(
            {
                "model": spec.name,
                "token_mode": spec.token_mode,
                "layers": spec.num_layers,
                "heads": spec.nhead,
                "d_model": spec.d_model,
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "train_samples": len(result["train_meta"]),
                "test_samples": len(result["test_meta"]),
                "train_unique_checkpoints": int(result["train_meta"][["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "test_unique_checkpoints": int(result["test_meta"][["battery", "assigned_checkpoint_cycle"]].drop_duplicates().shape[0]),
                "event_train_mae_pp": result["train_metrics_event"]["mae"],
                "event_train_rmse_pp": result["train_metrics_event"]["rmse"],
                "event_train_r2": result["train_metrics_event"]["r2"],
                "event_test_mae_pp": result["test_metrics_event"]["mae"],
                "event_test_rmse_pp": result["test_metrics_event"]["rmse"],
                "event_test_r2": result["test_metrics_event"]["r2"],
                "checkpoint_test_mae_pp": result["test_metrics_checkpoint"]["mae"],
                "checkpoint_test_rmse_pp": result["test_metrics_checkpoint"]["rmse"],
                "checkpoint_test_r2": result["test_metrics_checkpoint"]["r2"],
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("checkpoint_test_mae_pp").reset_index(drop=True)
    summary_df.to_csv(OUTDIR / "comparison_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(summary_df["model"], summary_df["event_test_r2"], color="#457b9d")
    axes[0].set_title("RW12 event-level R2")
    axes[0].set_ylabel("R2")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(summary_df["model"], summary_df["checkpoint_test_r2"], color="#e76f51")
    axes[1].set_title("RW12 checkpoint-aggregated R2")
    axes[1].set_ylabel("R2")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=15)
    fig.savefig(OUTDIR / "comparison_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    results: list[dict[str, Any]] = []
    for spec in SPECS:
        print(f"Running {spec.name} ...", flush=True)
        result = train_and_evaluate(spec)
        save_run_artifacts(result, OUTDIR / spec.name)
        results.append(result)
    save_comparison(results)


if __name__ == "__main__":
    main()
