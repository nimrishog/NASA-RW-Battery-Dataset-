from __future__ import annotations

import copy
import json
import math
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
from torch.utils.data import DataLoader, Dataset


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
PKL_DIR = BASE / "charge pkl with temperature"
FEATURE_DIR = BASE / "charge engineered features"
ANALYSIS_DIR = BASE / "Analysis"
OUTDIR = BASE / "clean_transformer_rebuild"

TRAIN_BATTERIES = ("RW9", "RW10")
VAL_BATTERY = "RW11"
FINAL_TRAIN_BATTERIES = ("RW9", "RW10", "RW11")
TEST_BATTERY = "RW12"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    option: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    learning_rate: float
    max_epochs: int
    patience: int
    token_count: int
    voltage_window: tuple[float, float] | None = None
    segment_history: int | None = None


SPECS = [
    ModelSpec("option1_last_segment_h4", "option1", 48, 4, 2, 96, 0.10, 5.0e-4, 120, 12, 48),
    ModelSpec("option1_last_segment_h8", "option1", 64, 8, 2, 128, 0.10, 5.0e-4, 120, 12, 48),
    ModelSpec("option2_history16_h4", "option2", 64, 4, 2, 128, 0.10, 5.0e-4, 160, 16, 16, segment_history=16),
    ModelSpec("option2_history16_h8", "option2", 64, 8, 2, 128, 0.10, 5.0e-4, 160, 16, 16, segment_history=16),
    ModelSpec("option3_v38_40_h4", "option3", 48, 4, 2, 96, 0.10, 5.0e-4, 120, 12, 32, voltage_window=(3.8, 4.0), segment_history=16),
    ModelSpec("option3_v38_40_h8", "option3", 64, 8, 2, 128, 0.10, 5.0e-4, 120, 12, 32, voltage_window=(3.8, 4.0), segment_history=16),
]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, mask: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
        self.x = torch.from_numpy(x).float()
        self.mask = torch.from_numpy(mask).bool()
        self.y = torch.from_numpy(y).float()
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        return (
            self.x[idx],
            self.mask[idx],
            self.y[idx],
            row["battery"],
            int(row["checkpoint_cycle"]),
            str(row["sample_id"]),
        )


class BenchmarkTransformer(nn.Module):
    def __init__(self, input_dim: int, spec: ModelSpec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.input_proj = nn.Linear(input_dim, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
        pe = create_sinusoidal_positional_encoding(spec.token_count + 1, spec.d_model)
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_sinusoidal_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


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


def load_reference_targets(battery: str) -> tuple[pd.DataFrame, float]:
    ref = pd.read_csv(ANALYSIS_DIR / battery / "reference_capacity_summary.csv")
    grouped = (
        ref.groupby("charge_cycle_count_before_reference", as_index=False)
        .agg(
            capacity_ah=("capacity_ah", "mean"),
            sample_count=("sample_count", "sum"),
            first_date=("date", "first"),
        )
        .sort_values("charge_cycle_count_before_reference")
        .reset_index(drop=True)
    )
    initial_capacity = float(grouped.loc[grouped["charge_cycle_count_before_reference"].eq(0), "capacity_ah"].mean())
    grouped["soh_percent"] = grouped["capacity_ah"] / initial_capacity * 100.0
    grouped["battery"] = battery
    grouped = grouped[grouped["charge_cycle_count_before_reference"] > 0].reset_index(drop=True)
    return grouped, initial_capacity


def load_charge_samples(battery: str) -> pd.DataFrame:
    return pd.read_pickle(PKL_DIR / f"{battery}_charge_temp_datacapa.pkl").sort_values(["cycle", "relTime", "time"]).reset_index(drop=True)


def load_charge_features(battery: str) -> pd.DataFrame:
    df = pd.read_csv(FEATURE_DIR / f"{battery}_charge_engineered_features.csv").sort_values("cycle").reset_index(drop=True)
    battery_time0 = float(df["time_start"].min())
    df["elapsed_days"] = (df["time_end"] - battery_time0) / 86400.0
    return df


def diff_with_zero(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.size > 1:
        out[1:] = np.diff(values)
    return out.astype(np.float32)


def cumulative_segment_charge(current: np.ndarray, rel_time: np.ndarray) -> np.ndarray:
    dt = np.clip(diff_with_zero(rel_time), 0.0, None)
    return np.cumsum((-current * dt) / 3600.0).astype(np.float32)


def cumulative_segment_energy(voltage: np.ndarray, current: np.ndarray, rel_time: np.ndarray) -> np.ndarray:
    dt = np.clip(diff_with_zero(rel_time), 0.0, None)
    return np.cumsum((voltage * (-current) * dt) / 3600.0).astype(np.float32)


def safe_ratio(num: np.ndarray, den: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    out = np.full_like(num, fallback, dtype=np.float32)
    valid = np.isfinite(den) & (np.abs(den) > 1.0e-8)
    out[valid] = (num[valid] / den[valid]).astype(np.float32)
    return out


def resample_matrix(seq: np.ndarray, token_count: int) -> np.ndarray:
    if len(seq) == token_count:
        return seq.astype(np.float32, copy=False)
    if len(seq) == 1:
        return np.repeat(seq.astype(np.float32, copy=False), token_count, axis=0)
    source_x = np.linspace(0.0, 1.0, len(seq), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, token_count, dtype=np.float32)
    out = np.zeros((token_count, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(target_x, source_x, seq[:, j]).astype(np.float32)
    return out


def monotonic_interp(x: np.ndarray, y: np.ndarray, target_x: np.ndarray) -> np.ndarray | None:
    if x.size < 2:
        return None
    mono_x = np.maximum.accumulate(x.astype(np.float64))
    unique_x, unique_idx = np.unique(mono_x, return_index=True)
    if unique_x.size < 2 or target_x[0] < unique_x[0] or target_x[-1] > unique_x[-1]:
        return None
    return np.interp(target_x, unique_x, y[unique_idx].astype(np.float64)).astype(np.float32)


def build_option1_samples(batteries: tuple[str, ...], spec: ModelSpec) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for battery in batteries:
        checkpoints, initial_capacity = load_reference_targets(battery)
        samples_df = load_charge_samples(battery)
        battery_time0 = float(samples_df["time"].min())
        grouped = {int(cycle): group for cycle, group in samples_df.groupby("cycle", sort=True)}

        for row in checkpoints.itertuples(index=False):
            cycle = int(row.charge_cycle_count_before_reference)
            if cycle not in grouped:
                continue
            group = grouped[cycle]
            voltage = group["Voltage"].to_numpy(dtype=np.float32)
            current = group["Current"].to_numpy(dtype=np.float32)
            temperature = group["Temperature"].to_numpy(dtype=np.float32)
            rel_time = group["relTime"].to_numpy(dtype=np.float32)
            global_days = ((group["time"].to_numpy(dtype=np.float64) - battery_time0) / 86400.0).astype(np.float32)

            delta_t = np.clip(diff_with_zero(rel_time), 0.0, None)
            cumulative_charge = cumulative_segment_charge(current, rel_time)
            cumulative_energy = cumulative_segment_energy(voltage, current, rel_time)
            delta_v = diff_with_zero(voltage)
            delta_temp = diff_with_zero(temperature)
            dv_dt = safe_ratio(delta_v, delta_t)
            dtemp_dt = safe_ratio(delta_temp, delta_t)
            voltage_position = np.clip((voltage - 3.2) / (4.2 - 3.2), 0.0, 1.0).astype(np.float32)
            dq = diff_with_zero(cumulative_charge)
            dq_dv = safe_ratio(dq, np.where(np.abs(delta_v) < 1.0e-6, np.nan, delta_v))
            current_rate = (current / initial_capacity).astype(np.float32)

            seq = np.column_stack(
                [
                    voltage,
                    current,
                    temperature,
                    rel_time,
                    global_days,
                    delta_t,
                    cumulative_charge,
                    cumulative_energy,
                    delta_v,
                    delta_temp,
                    dv_dt,
                    dtemp_dt,
                    voltage_position,
                    dq_dv,
                    current_rate,
                ]
            ).astype(np.float32)
            x_rows.append(resample_matrix(seq, spec.token_count))
            mask_rows.append(np.ones(spec.token_count, dtype=bool))
            meta_rows.append(
                {
                    "battery": battery,
                    "checkpoint_cycle": cycle,
                    "sample_id": f"{battery}_cp{cycle}_lastsegment",
                    "target_soh_percent": float(row.soh_percent),
                    "target_capacity_ah": float(row.capacity_ah),
                    "option": spec.option,
                }
            )

    return np.stack(x_rows), np.stack(mask_rows), pd.DataFrame(meta_rows)


def build_option2_samples(batteries: tuple[str, ...], spec: ModelSpec) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    history = int(spec.segment_history or spec.token_count)
    token_cols = [
        "duration_s",
        "sample_count",
        "charge_throughput_ah",
        "energy_throughput_wh",
        "power_mean_w",
        "power_max_w",
        "voltage_start",
        "voltage_end",
        "voltage_delta",
        "voltage_slope_vps",
        "dv_dt_mean_vps",
        "current_mean_a",
        "current_abs_mean_a",
        "current_std_a",
        "temperature_start_c",
        "temperature_end_c",
        "temperature_delta_c",
        "temperature_mean_c",
        "temperature_std_c",
        "dtemp_dt_mean_cps",
        "time_in_vwin_3p8_4p0_s",
        "time_in_vwin_4p0_4p1_s",
        "time_cross_3p9_to_4p1_s",
        "dq_dv_median_ahpv",
        "dq_dv_max_ahpv",
        "dq_dv_min_ahpv",
        "dv_dq_median_vpah",
        "dv_dq_max_vpah",
        "dv_dq_min_vpah",
        "voltage_pdf_mean",
        "voltage_pdf_std",
        "temperature_pdf_mean",
        "temperature_pdf_std",
        "elapsed_days",
    ]

    x_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for battery in batteries:
        checkpoints, _ = load_reference_targets(battery)
        features = load_charge_features(battery)

        for row in checkpoints.itertuples(index=False):
            cycle = int(row.charge_cycle_count_before_reference)
            history_df = features[features["cycle"].le(cycle)].tail(history)
            token_matrix = history_df.loc[:, token_cols].to_numpy(dtype=np.float32, copy=False)
            padded = np.zeros((history, len(token_cols)), dtype=np.float32)
            mask = np.zeros(history, dtype=bool)
            padded[-len(history_df) :, :] = token_matrix
            mask[-len(history_df) :] = True
            x_rows.append(padded)
            mask_rows.append(mask)
            meta_rows.append(
                {
                    "battery": battery,
                    "checkpoint_cycle": cycle,
                    "sample_id": f"{battery}_cp{cycle}_hist{history}",
                    "target_soh_percent": float(row.soh_percent),
                    "target_capacity_ah": float(row.capacity_ah),
                    "option": spec.option,
                }
            )

    return np.stack(x_rows), np.stack(mask_rows), pd.DataFrame(meta_rows)


def build_option3_samples(batteries: tuple[str, ...], spec: ModelSpec) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if spec.voltage_window is None:
        raise ValueError("Option 3 requires a voltage window.")
    v_low, v_high = spec.voltage_window
    search_back = int(spec.segment_history or 1)
    grid = np.linspace(v_low, v_high, spec.token_count, dtype=np.float32)
    x_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for battery in batteries:
        checkpoints, initial_capacity = load_reference_targets(battery)
        samples_df = load_charge_samples(battery)
        battery_time0 = float(samples_df["time"].min())
        grouped = {int(cycle): group for cycle, group in samples_df.groupby("cycle", sort=True)}

        for row in checkpoints.itertuples(index=False):
            cycle = int(row.charge_cycle_count_before_reference)
            chosen_group = None
            chosen_cycle = None
            for candidate_cycle in range(cycle, max(0, cycle - search_back), -1):
                group = grouped.get(candidate_cycle)
                if group is None:
                    continue
                voltage = group["Voltage"].to_numpy(dtype=np.float32)
                if float(voltage.min()) <= v_low and float(voltage.max()) >= v_high:
                    chosen_group = group
                    chosen_cycle = candidate_cycle
                    break
            if chosen_group is None or chosen_cycle is None:
                continue

            group = chosen_group
            voltage = group["Voltage"].to_numpy(dtype=np.float32)
            current = group["Current"].to_numpy(dtype=np.float32)
            temperature = group["Temperature"].to_numpy(dtype=np.float32)
            rel_time = group["relTime"].to_numpy(dtype=np.float32)
            global_days = ((group["time"].to_numpy(dtype=np.float64) - battery_time0) / 86400.0).astype(np.float32)

            cumulative_charge = cumulative_segment_charge(current, rel_time)
            cumulative_energy = cumulative_segment_energy(voltage, current, rel_time)
            current_rate = (current / initial_capacity).astype(np.float32)

            current_grid = monotonic_interp(voltage, current, grid)
            temp_grid = monotonic_interp(voltage, temperature, grid)
            rel_grid = monotonic_interp(voltage, rel_time, grid)
            days_grid = monotonic_interp(voltage, global_days, grid)
            charge_grid = monotonic_interp(voltage, cumulative_charge, grid)
            energy_grid = monotonic_interp(voltage, cumulative_energy, grid)
            rate_grid = monotonic_interp(voltage, current_rate, grid)
            if any(item is None for item in (current_grid, temp_grid, rel_grid, days_grid, charge_grid, energy_grid, rate_grid)):
                continue

            delta_t = np.clip(diff_with_zero(rel_grid), 0.0, None)
            delta_temp = diff_with_zero(temp_grid)
            delta_v = diff_with_zero(grid)
            dq = diff_with_zero(charge_grid)
            dq_dv = safe_ratio(dq, np.where(np.abs(delta_v) < 1.0e-6, np.nan, delta_v))
            dtemp_dt = safe_ratio(delta_temp, delta_t)
            voltage_pos = ((grid - v_low) / max(v_high - v_low, 1.0e-8)).astype(np.float32)

            seq = np.column_stack(
                [
                    grid.astype(np.float32),
                    current_grid,
                    temp_grid,
                    rel_grid,
                    days_grid,
                    delta_t,
                    charge_grid,
                    energy_grid,
                    delta_temp,
                    dtemp_dt,
                    voltage_pos,
                    dq_dv,
                    rate_grid,
                ]
            ).astype(np.float32)
            x_rows.append(seq)
            mask_rows.append(np.ones(spec.token_count, dtype=bool))
            meta_rows.append(
                {
                    "battery": battery,
                    "checkpoint_cycle": cycle,
                    "sample_id": f"{battery}_cp{cycle}_vwin_from_{chosen_cycle}",
                    "target_soh_percent": float(row.soh_percent),
                    "target_capacity_ah": float(row.capacity_ah),
                    "option": spec.option,
                }
            )

    return np.stack(x_rows), np.stack(mask_rows), pd.DataFrame(meta_rows)


def build_dataset_for_batteries(spec: ModelSpec, batteries: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if spec.option == "option1":
        return build_option1_samples(batteries, spec)
    if spec.option == "option2":
        return build_option2_samples(batteries, spec)
    if spec.option == "option3":
        return build_option3_samples(batteries, spec)
    raise ValueError(f"Unsupported option: {spec.option}")


def sanitize_and_standardize(
    train_x: np.ndarray,
    other_x: np.ndarray,
    train_mask: np.ndarray,
    other_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    valid_train = train_x[train_mask]
    low = np.nanpercentile(valid_train, 0.5, axis=0)
    high = np.nanpercentile(valid_train, 99.5, axis=0)
    train_x = np.nan_to_num(np.clip(train_x, low, high), nan=0.0, posinf=0.0, neginf=0.0)
    other_x = np.nan_to_num(np.clip(other_x, low, high), nan=0.0, posinf=0.0, neginf=0.0)
    valid_train = train_x[train_mask]
    mean = valid_train.mean(axis=0)
    std = valid_train.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    train_x = ((train_x - mean) / std).astype(np.float32)
    other_x = ((other_x - mean) / std).astype(np.float32)
    return train_x, other_x, {"clip_low": low, "clip_high": high, "mean": mean, "std": std}


def standardize_target(train_y: np.ndarray, other_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((other_y - mean) / std).astype(np.float32), mean, std


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[int], list[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batteries: list[str] = []
    checkpoint_cycles: list[int] = []
    sample_ids: list[str] = []

    with torch.set_grad_enabled(training):
        for batch_x, batch_mask, batch_y, batch_battery, batch_checkpoint, batch_sample_id in loader:
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

            bs = batch_x.size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
            preds.append(out.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())
            batteries.extend(list(batch_battery))
            checkpoint_cycles.extend([int(x) for x in batch_checkpoint])
            sample_ids.extend(list(batch_sample_id))

    return total_loss / max(total_count, 1), np.concatenate(preds), np.concatenate(targets), batteries, checkpoint_cycles, sample_ids


def destandardize(y_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_norm * std + mean


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_loss_curves(history: pd.DataFrame, outdir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train", linewidth=2)
    ax.plot(history["epoch"], history["val_loss"], label="validation", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on standardized SOH")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "loss_curves.png", dpi=180)
    plt.close(fig)


def save_predictions_plot(pred_df: pd.DataFrame, outdir: Path, title_prefix: str) -> None:
    y_true = pred_df["actual_soh_percent"].to_numpy(dtype=np.float64)
    y_pred = pred_df["predicted_soh_percent"].to_numpy(dtype=np.float64)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(y_true, y_pred, s=20, alpha=0.6)
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[0].set_title(f"{title_prefix}: predicted vs actual SOH")
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")
    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, s=20, alpha=0.6)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title(f"{title_prefix}: residuals")
    axes[1].set_xlabel("Actual SOH (%)")
    axes[1].set_ylabel("Error (pp)")
    fig.savefig(outdir / "soh_predictions.png", dpi=180)
    plt.close(fig)


def train_with_validation(spec: ModelSpec, train_batteries: tuple[str, ...], val_battery: str) -> dict[str, object]:
    train_x, train_mask, train_meta = build_dataset_for_batteries(spec, train_batteries)
    val_x, val_mask, val_meta = build_dataset_for_batteries(spec, (val_battery,))
    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    val_y = val_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_x, val_x, feature_stats = sanitize_and_standardize(train_x, val_x, train_mask, val_mask)
    train_y_norm, val_y_norm, y_mean, y_std = standardize_target(train_y, val_y)

    train_ds = SequenceDataset(train_x, train_mask, train_y_norm, train_meta)
    val_ds = SequenceDataset(val_x, val_mask, val_y_norm, val_meta)
    train_loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(32, len(val_ds)), shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = BenchmarkTransformer(train_x.shape[-1], spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=1.0e-4)

    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, spec.max_epochs + 1):
        train_loss, train_pred_norm, train_true_norm, *_ = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_pred_norm, val_true_norm, *_ = run_epoch(model, val_loader, None, device)

        train_pred = destandardize(train_pred_norm, y_mean, y_std)
        train_true = destandardize(train_true_norm, y_mean, y_std)
        val_pred = destandardize(val_pred_norm, y_mean, y_std)
        val_true = destandardize(val_true_norm, y_mean, y_std)
        train_metrics = regression_metrics(train_true, train_pred)
        val_metrics = regression_metrics(val_true, val_pred)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= spec.patience:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint for {spec.name}")

    model.load_state_dict(best_state)
    _, val_pred_norm, val_true_norm, val_battery_ids, val_cycles, val_ids = run_epoch(model, val_loader, None, device)
    val_pred = destandardize(val_pred_norm, y_mean, y_std)
    val_true = destandardize(val_true_norm, y_mean, y_std)

    return {
        "spec": spec,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "validation_metrics": regression_metrics(val_true, val_pred),
        "history": pd.DataFrame(history_rows),
        "feature_stats": feature_stats,
        "target_mean": y_mean,
        "target_std": y_std,
        "parameter_count": parameter_count(model),
        "validation_predictions": pd.DataFrame(
            {
                "battery": val_battery_ids,
                "checkpoint_cycle": val_cycles,
                "sample_id": val_ids,
                "actual_soh_percent": val_true,
                "predicted_soh_percent": val_pred,
                "absolute_error_pp": np.abs(val_pred - val_true),
            }
        ),
    }


def retrain_and_test(spec: ModelSpec, best_epoch: int) -> dict[str, object]:
    train_x, train_mask, train_meta = build_dataset_for_batteries(spec, FINAL_TRAIN_BATTERIES)
    test_x, test_mask, test_meta = build_dataset_for_batteries(spec, (TEST_BATTERY,))
    train_y = train_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["target_soh_percent"].to_numpy(dtype=np.float32)
    train_x, test_x, feature_stats = sanitize_and_standardize(train_x, test_x, train_mask, test_mask)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    train_ds = SequenceDataset(train_x, train_mask, train_y_norm, train_meta)
    test_ds = SequenceDataset(test_x, test_mask, test_y_norm, test_meta)
    train_loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=min(32, len(test_ds)), shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = BenchmarkTransformer(train_x.shape[-1], spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=1.0e-4)
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, best_epoch + 1):
        train_loss, train_pred_norm, train_true_norm, *_ = run_epoch(model, train_loader, optimizer, device)
        train_pred = destandardize(train_pred_norm, y_mean, y_std)
        train_true = destandardize(train_true_norm, y_mean, y_std)
        train_metrics = regression_metrics(train_true, train_pred)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
            }
        )

    _, train_pred_norm, train_true_norm, train_battery_ids, train_cycles, train_ids = run_epoch(model, train_loader, None, device)
    _, test_pred_norm, test_true_norm, test_battery_ids, test_cycles, test_ids = run_epoch(model, test_loader, None, device)
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)

    return {
        "feature_stats": feature_stats,
        "target_mean": y_mean,
        "target_std": y_std,
        "train_metrics": regression_metrics(train_true, train_pred),
        "test_metrics": regression_metrics(test_true, test_pred),
        "train_predictions": pd.DataFrame(
            {
                "battery": train_battery_ids,
                "checkpoint_cycle": train_cycles,
                "sample_id": train_ids,
                "actual_soh_percent": train_true,
                "predicted_soh_percent": train_pred,
                "absolute_error_pp": np.abs(train_pred - train_true),
            }
        ),
        "test_predictions": pd.DataFrame(
            {
                "battery": test_battery_ids,
                "checkpoint_cycle": test_cycles,
                "sample_id": test_ids,
                "actual_soh_percent": test_true,
                "predicted_soh_percent": test_pred,
                "absolute_error_pp": np.abs(test_pred - test_true),
            }
        ),
        "history": pd.DataFrame(history_rows),
        "parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "state_dict": copy.deepcopy(model.state_dict()),
    }


def save_selection_and_final_artifacts(selection: dict[str, object], final_run: dict[str, object], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    spec = selection["spec"]
    selection["history"].to_csv(outdir / "selection_history.csv", index=False)
    save_loss_curves(selection["history"], outdir, f"{spec.name}: validation tuning")
    selection["validation_predictions"].to_csv(outdir / "validation_predictions.csv", index=False)
    save_predictions_plot(selection["validation_predictions"], outdir, f"{spec.name} validation")

    final_run["history"].to_csv(outdir / "final_train_history.csv", index=False)
    final_run["train_predictions"].to_csv(outdir / "train_predictions.csv", index=False)
    final_run["test_predictions"].to_csv(outdir / "test_predictions.csv", index=False)
    save_predictions_plot(final_run["test_predictions"], outdir, f"{spec.name} RW12 test")
    torch.save(final_run["state_dict"], outdir / "best_model.pt")

    metrics_df = pd.DataFrame(
        [
            {"split": "selection_validation", **selection["validation_metrics"]},
            {"split": "final_train", **final_run["train_metrics"]},
            {"split": "final_test_rw12", **final_run["test_metrics"]},
        ]
    )
    metrics_df.to_csv(outdir / "metrics.csv", index=False)

    summary = {
        "spec": asdict(spec),
        "parameter_count": final_run["parameter_count"],
        "best_epoch_from_validation": selection["best_epoch"],
        "validation_metrics": selection["validation_metrics"],
        "final_train_metrics": final_run["train_metrics"],
        "final_test_metrics": final_run["test_metrics"],
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    selection_rows: list[dict[str, object]] = []
    results_rows: list[dict[str, object]] = []
    best_spec_name = None
    best_val_mae = float("inf")

    for spec in SPECS:
        print(f"Tuning {spec.name} ...", flush=True)
        selection = train_with_validation(spec, TRAIN_BATTERIES, VAL_BATTERY)
        final_run = retrain_and_test(spec, int(selection["best_epoch"]))
        combo_dir = OUTDIR / spec.name
        save_selection_and_final_artifacts(selection, final_run, combo_dir)

        selection_rows.append(
            {
                "model": spec.name,
                "option": spec.option,
                "heads": spec.nhead,
                "layers": spec.num_layers,
                "d_model": spec.d_model,
                "token_count": spec.token_count,
                "best_epoch": selection["best_epoch"],
                "val_mae_pp": selection["validation_metrics"]["mae"],
                "val_rmse_pp": selection["validation_metrics"]["rmse"],
                "val_r2": selection["validation_metrics"]["r2"],
            }
        )
        results_rows.append(
            {
                "model": spec.name,
                "option": spec.option,
                "heads": spec.nhead,
                "layers": spec.num_layers,
                "d_model": spec.d_model,
                "token_count": spec.token_count,
                "best_epoch": selection["best_epoch"],
                "train_mae_pp": final_run["train_metrics"]["mae"],
                "train_rmse_pp": final_run["train_metrics"]["rmse"],
                "train_r2": final_run["train_metrics"]["r2"],
                "test_mae_pp": final_run["test_metrics"]["mae"],
                "test_rmse_pp": final_run["test_metrics"]["rmse"],
                "test_r2": final_run["test_metrics"]["r2"],
                "test_mape_percent": final_run["test_metrics"]["mape_percent"],
            }
        )

        if selection["validation_metrics"]["mae"] < best_val_mae:
            best_val_mae = selection["validation_metrics"]["mae"]
            best_spec_name = spec.name

    selection_df = pd.DataFrame(selection_rows).sort_values("val_mae_pp").reset_index(drop=True)
    selection_df.to_csv(OUTDIR / "selection_summary.csv", index=False)
    results_df = pd.DataFrame(results_rows).sort_values("test_mae_pp").reset_index(drop=True)
    results_df.to_csv(OUTDIR / "test_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(selection_df["model"], selection_df["val_mae_pp"], color="#457b9d")
    axes[0].invert_yaxis()
    axes[0].set_title("Validation MAE by model")
    axes[0].set_xlabel("MAE (SOH percentage points)")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(results_df["model"], results_df["test_mae_pp"], color="#e76f51")
    axes[1].invert_yaxis()
    axes[1].set_title("RW12 test MAE by model")
    axes[1].set_xlabel("MAE (SOH percentage points)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(OUTDIR / "comparison_summary.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# Clean Benchmark-Aligned Transformer Rebuild",
        "",
        "Protocol:",
        "- Validation tuning: train on RW9 + RW10, validate on RW11",
        "- Final training: retrain on full RW9 + RW10 + RW11",
        "- Test: full RW12",
        "- Target: SOH = benchmark capacity / initial capacity * 100",
        "",
        "Options:",
        "- Option 1: last partial charge segment before each benchmark",
        "- Option 2: last N partial charge segments before each benchmark",
        "- Option 3: fixed voltage-window sequence from the last partial charge segment before each benchmark",
        "",
        f"Best validation model: {best_spec_name}",
        "",
        "Test summary (sorted by MAE):",
    ]
    for _, row in results_df.iterrows():
        report_lines.append(
            f"- {row['model']}: test MAE={row['test_mae_pp']:.4f} pp, "
            f"RMSE={row['test_rmse_pp']:.4f} pp, R2={row['test_r2']:.4f}"
        )
    (OUTDIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
