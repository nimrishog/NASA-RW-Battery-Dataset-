from __future__ import annotations

import gc
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
PKL_DIR = BASE / "charge pkl with temperature"
ANALYSIS_DIR = BASE / "Analysis"
OUTDIR = BASE / "dense_sample_transformers"

TRAIN_BATTERIES = ("RW9", "RW10", "RW11")
TEST_BATTERIES = ("RW12",)
PATCH_SIZE = 30

ROW_FEATURES = [
    "Voltage",
    "Current",
    "Temperature",
    "relTime",
    "delta_t",
    "cumulative_charge_ah",
    "cumulative_energy_wh",
    "delta_v",
    "delta_temperature",
    "elapsed_days",
    "voltage_position",
    "current_rate",
]


@dataclass(frozen=True)
class SampleSpec:
    name: str
    mode: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int


PATCH_SPEC = SampleSpec("patch30_rows_deep", "patch30", 64, 8, 4, 128, 0.10, 3.0e-4, 1.0e-4, 4, 2048)
ROW_SPEC = SampleSpec("single_row_feature_transformer", "row", 32, 4, 3, 64, 0.10, 3.0e-4, 1.0e-4, 2, 32768)


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


class SequenceTransformer(nn.Module):
    def __init__(self, input_dim: int, max_tokens: int, spec: SampleSpec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.input_proj = nn.Linear(input_dim, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        pe = create_sinusoidal_positional_encoding(max_tokens + 1, spec.d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.nhead,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
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


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, spec: SampleSpec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.scalar_proj = nn.Linear(1, spec.d_model)
        self.feature_embedding = nn.Parameter(torch.zeros(1, input_dim, spec.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.nhead,
            dim_feedforward=spec.dim_feedforward,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(spec.d_model),
            nn.Linear(spec.d_model, spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.scalar_proj(x.unsqueeze(-1)) + self.feature_embedding
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0]).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def next_benchmark_label_map(battery: str, max_cycle: int) -> tuple[dict[int, float], dict[int, int], float]:
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
    cycle_to_checkpoint: dict[int, int] = {}
    for cycle in range(1, max_cycle + 1):
        idx = int(np.searchsorted(checkpoint_cycles, cycle, side="left"))
        if idx >= len(checkpoint_cycles):
            idx = len(checkpoint_cycles) - 1
        cycle_to_soh[cycle] = float(soh_values[idx])
        cycle_to_checkpoint[cycle] = int(checkpoint_cycles[idx])
    return cycle_to_soh, cycle_to_checkpoint, initial_capacity


def diff_with_zero(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        out[1:] = np.diff(values).astype(np.float32)
    return out


def build_battery_rows(battery: str) -> dict[str, np.ndarray]:
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
    cycle_to_soh, cycle_to_checkpoint, initial_capacity = next_benchmark_label_map(battery, int(unique_cycles.max()))
    battery_time0 = float(abs_time.min())

    x = np.zeros((len(df), len(ROW_FEATURES)), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.float32)
    checkpoints = np.zeros(len(df), dtype=np.int32)

    for cycle, start, end in zip(unique_cycles.tolist(), starts.tolist(), ends.tolist()):
        sl = slice(start, end)
        v = voltage[sl]
        c = current[sl]
        t = temperature[sl]
        rt = rel_time[sl]
        at = abs_time[sl]
        dt = np.clip(diff_with_zero(rt), 0.0, None)
        cumulative_charge = np.cumsum((-c * dt) / 3600.0, dtype=np.float32)
        cumulative_energy = np.cumsum((v * (-c) * dt) / 3600.0, dtype=np.float32)
        delta_v = diff_with_zero(v)
        delta_temp = diff_with_zero(t)
        elapsed_days = ((at - battery_time0) / 86400.0).astype(np.float32)
        voltage_position = np.clip((v - 3.2) / 1.0, 0.0, 1.0).astype(np.float32)
        current_rate = (c / max(initial_capacity, 1.0e-8)).astype(np.float32)
        x[sl, :] = np.column_stack(
            [v, c, t, rt, dt, cumulative_charge, cumulative_energy, delta_v, delta_temp, elapsed_days, voltage_position, current_rate]
        )
        y[sl] = cycle_to_soh[int(cycle)]
        checkpoints[sl] = cycle_to_checkpoint[int(cycle)]

    return {
        "x": np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        "y": y,
        "cycle": cycles,
        "cycle_key": cycles.astype(np.int64),
        "checkpoint": checkpoints,
    }


def concat_rows(batteries: tuple[str, ...]) -> dict[str, np.ndarray]:
    parts = []
    for battery_idx, battery in enumerate(batteries):
        part = build_battery_rows(battery)
        # Cycle ids restart for every battery, so make a unique key before concatenation.
        part["cycle_key"] = part["cycle"].astype(np.int64) + battery_idx * 1_000_000
        parts.append(part)
    return {
        "x": np.concatenate([part["x"] for part in parts], axis=0),
        "y": np.concatenate([part["y"] for part in parts], axis=0),
        "cycle": np.concatenate([part["cycle"] for part in parts], axis=0),
        "cycle_key": np.concatenate([part["cycle_key"] for part in parts], axis=0),
        "checkpoint": np.concatenate([part["checkpoint"] for part in parts], axis=0),
    }


def standardize_in_place(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    train_x -= mean
    train_x /= std
    test_x -= mean
    test_x /= std
    return mean, std


def standardize_targets(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((test_y - mean) / std).astype(np.float32), mean, std


def build_patch_arrays(table: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = table["x"]
    y = table["y"]
    cycles = table["cycle_key"] if "cycle_key" in table else table["cycle"]
    checkpoints = table["checkpoint"]
    unique_cycles, starts = np.unique(cycles, return_index=True)
    ends = np.r_[starts[1:], len(cycles)]
    total_patches = int(sum(math.ceil((end - start) / PATCH_SIZE) for start, end in zip(starts, ends)))
    patch_x = np.zeros((total_patches, PATCH_SIZE, x.shape[1]), dtype=np.float32)
    patch_mask = np.zeros((total_patches, PATCH_SIZE), dtype=bool)
    patch_y = np.zeros(total_patches, dtype=np.float32)
    patch_checkpoint = np.zeros(total_patches, dtype=np.int32)

    row = 0
    for start, end in zip(starts.tolist(), ends.tolist()):
        for patch_start in range(start, end, PATCH_SIZE):
            patch_end = min(patch_start + PATCH_SIZE, end)
            length = patch_end - patch_start
            patch_x[row, :length, :] = x[patch_start:patch_end]
            patch_mask[row, :length] = True
            patch_y[row] = y[patch_start]
            patch_checkpoint[row] = checkpoints[patch_start]
            row += 1
    return patch_x, patch_mask, patch_y, patch_checkpoint


def train_sequence_model(
    spec: SampleSpec,
    train_x: np.ndarray,
    train_mask: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_mask: np.ndarray,
    test_y: np.ndarray,
) -> tuple[nn.Module, pd.DataFrame]:
    model = SequenceTransformer(train_x.shape[-1], PATCH_SIZE, spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_mask), torch.from_numpy(train_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_mask), torch.from_numpy(test_y))
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, spec.epochs + 1):
        train_loss = sequence_epoch(model, train_loader, optimizer)
        test_loss = sequence_epoch(model, test_loader, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(f"{spec.name} epoch {epoch}/{spec.epochs}: train_loss={train_loss:.5f} test_loss={test_loss:.5f}", flush=True)
    return model, pd.DataFrame(history)


def sequence_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    with torch.set_grad_enabled(training):
        for batch_x, batch_mask, batch_y in loader:
            out = model(batch_x.float(), batch_mask.bool())
            loss = nn.functional.mse_loss(out, batch_y.float())
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_count += len(batch_y)
    return total_loss / max(total_count, 1)


def predict_sequence(model: nn.Module, x: np.ndarray, mask: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(mask)), batch_size=batch_size, shuffle=False, num_workers=0)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_mask in loader:
            preds.append(model(batch_x.float(), batch_mask.bool()).cpu().numpy())
    return np.concatenate(preds)


def train_feature_model(
    spec: SampleSpec,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> tuple[nn.Module, pd.DataFrame]:
    model = FeatureTransformer(train_x.shape[1], spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, spec.epochs + 1):
        train_loss = feature_epoch(model, train_loader, optimizer)
        test_loss = feature_epoch(model, test_loader, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(f"{spec.name} epoch {epoch}/{spec.epochs}: train_loss={train_loss:.5f} test_loss={test_loss:.5f}", flush=True)
    return model, pd.DataFrame(history)


def feature_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    with torch.set_grad_enabled(training):
        for batch_x, batch_y in loader:
            out = model(batch_x.float())
            loss = nn.functional.mse_loss(out, batch_y.float())
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_count += len(batch_y)
    return total_loss / max(total_count, 1)


def predict_feature(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False, num_workers=0)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            preds.append(model(batch_x.float()).cpu().numpy())
    return np.concatenate(preds)


def checkpoint_metrics(y_true: np.ndarray, y_pred: np.ndarray, checkpoints: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    df = pd.DataFrame({"checkpoint": checkpoints, "actual": y_true, "predicted": y_pred})
    grouped = df.groupby("checkpoint", as_index=False).agg(actual=("actual", "mean"), predicted=("predicted", "mean"), n=("actual", "size"))
    return regression_metrics(grouped["actual"].to_numpy(), grouped["predicted"].to_numpy()), grouped


def save_loss_plot(history: pd.DataFrame, outdir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train", linewidth=2)
    ax.plot(history["epoch"], history["test_loss"], label="test", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on standardized SOH")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "loss_curves.png", dpi=180)
    plt.close(fig)


def save_prediction_plots(y_true: np.ndarray, y_pred: np.ndarray, checkpoint_df: pd.DataFrame, outdir: Path, title: str) -> None:
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sample = np.linspace(0, len(y_true) - 1, min(len(y_true), 50000)).astype(int)
    axes[0].scatter(y_true[sample], y_pred[sample], s=6, alpha=0.15)
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[0].set_title(f"{title}: sampled predictions")
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")
    axes[1].plot(checkpoint_df["checkpoint"], checkpoint_df["actual"], marker="o", label="actual")
    axes[1].plot(checkpoint_df["checkpoint"], checkpoint_df["predicted"], marker="o", label="predicted")
    axes[1].set_title(f"{title}: checkpoint aggregation")
    axes[1].set_xlabel("Assigned benchmark checkpoint")
    axes[1].set_ylabel("SOH (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(outdir / "predictions.png", dpi=180)
    plt.close(fig)


def save_run(
    spec: SampleSpec,
    history: pd.DataFrame,
    model: nn.Module,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    checkpoints: np.ndarray,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    checkpoint_metric: dict[str, float],
    checkpoint_df: pd.DataFrame,
    train_count: int,
    test_count: int,
) -> None:
    outdir = OUTDIR / spec.name
    outdir.mkdir(parents=True, exist_ok=True)
    history.to_csv(outdir / "training_history.csv", index=False)
    checkpoint_df.to_csv(outdir / "test_checkpoint_predictions.csv", index=False)
    pred_sample = pd.DataFrame({"actual_soh_percent": y_true, "predicted_soh_percent": y_pred, "checkpoint": checkpoints})
    if len(pred_sample) > 250000:
        pred_sample = pred_sample.sample(250000, random_state=42).sort_index()
    pred_sample.to_csv(outdir / "test_predictions_sample.csv", index=False)
    save_loss_plot(history, outdir, f"{spec.name}: train/test loss")
    save_prediction_plots(y_true, y_pred, checkpoint_df, outdir, spec.name)
    torch.save(model.state_dict(), outdir / "model.pt")
    summary = {
        "spec": asdict(spec),
        "parameter_count": parameter_count(model),
        "train_samples": train_count,
        "test_samples": test_count,
        "test_unique_checkpoints": int(pd.Series(checkpoints).nunique()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "checkpoint_aggregated_test_metrics": checkpoint_metric,
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    print("Building row-level train/test arrays ...", flush=True)
    train_rows = concat_rows(TRAIN_BATTERIES)
    test_rows = concat_rows(TEST_BATTERIES)
    mean, std = standardize_in_place(train_rows["x"], test_rows["x"])
    train_y_norm, test_y_norm, y_mean, y_std = standardize_targets(train_rows["y"], test_rows["y"])

    summary_rows: list[dict[str, float | int | str]] = []

    print("Building 30-row patches ...", flush=True)
    patch_train_x, patch_train_mask, patch_train_y, _ = build_patch_arrays(
        {"x": train_rows["x"], "y": train_y_norm, "cycle_key": train_rows["cycle_key"], "checkpoint": train_rows["checkpoint"]}
    )
    patch_test_x, patch_test_mask, patch_test_y, patch_test_checkpoint = build_patch_arrays(
        {"x": test_rows["x"], "y": test_y_norm, "cycle_key": test_rows["cycle_key"], "checkpoint": test_rows["checkpoint"]}
    )
    model, history = train_sequence_model(PATCH_SPEC, patch_train_x, patch_train_mask, patch_train_y, patch_test_x, patch_test_mask, patch_test_y)
    train_pred = predict_sequence(model, patch_train_x, patch_train_mask, PATCH_SPEC.batch_size) * y_std + y_mean
    test_pred = predict_sequence(model, patch_test_x, patch_test_mask, PATCH_SPEC.batch_size) * y_std + y_mean
    train_true = patch_train_y * y_std + y_mean
    test_true = patch_test_y * y_std + y_mean
    train_metrics = regression_metrics(train_true, train_pred)
    test_metrics = regression_metrics(test_true, test_pred)
    cp_metrics, cp_df = checkpoint_metrics(test_true, test_pred, patch_test_checkpoint)
    save_run(PATCH_SPEC, history, model, test_true, test_pred, patch_test_checkpoint, train_metrics, test_metrics, cp_metrics, cp_df, len(patch_train_y), len(patch_test_y))
    summary_rows.append({"model": PATCH_SPEC.name, "train_samples": len(patch_train_y), "test_samples": len(patch_test_y), "test_r2": test_metrics["r2"], "checkpoint_r2": cp_metrics["r2"], "test_mae_pp": test_metrics["mae"], "checkpoint_mae_pp": cp_metrics["mae"]})
    del patch_train_x, patch_train_mask, patch_train_y, patch_test_x, patch_test_mask, patch_test_y, model
    gc.collect()

    print("Training single-row feature transformer ...", flush=True)
    model, history = train_feature_model(ROW_SPEC, train_rows["x"], train_y_norm, test_rows["x"], test_y_norm)
    train_pred = predict_feature(model, train_rows["x"], ROW_SPEC.batch_size) * y_std + y_mean
    test_pred = predict_feature(model, test_rows["x"], ROW_SPEC.batch_size) * y_std + y_mean
    train_metrics = regression_metrics(train_rows["y"], train_pred)
    test_metrics = regression_metrics(test_rows["y"], test_pred)
    cp_metrics, cp_df = checkpoint_metrics(test_rows["y"], test_pred, test_rows["checkpoint"])
    save_run(ROW_SPEC, history, model, test_rows["y"], test_pred, test_rows["checkpoint"], train_metrics, test_metrics, cp_metrics, cp_df, len(train_rows["y"]), len(test_rows["y"]))
    summary_rows.append({"model": ROW_SPEC.name, "train_samples": len(train_rows["y"]), "test_samples": len(test_rows["y"]), "test_r2": test_metrics["r2"], "checkpoint_r2": cp_metrics["r2"], "test_mae_pp": test_metrics["mae"], "checkpoint_mae_pp": cp_metrics["mae"]})

    pd.DataFrame(summary_rows).to_csv(OUTDIR / "comparison_summary.csv", index=False)
    (OUTDIR / "feature_standardization.json").write_text(
        json.dumps({"features": ROW_FEATURES, "mean": mean.tolist(), "std": std.tolist(), "target_mean": y_mean, "target_std": y_std}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
