from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_charge_discharge_weighted_transformers import add_block_weights, load_reference_label_table, assign_next_checkpoint
from run_full_event_token_transformers import build_patch_sequence


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
SEGMENT_DIR = ROOT / "charge_discharge_weighted_transformer" / "extracted_segments"
OUTDIR = ROOT / "charge_discharge_full_cycle_suite"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)
BATTERIES = (*TRAIN_BATTERIES, *TEST_BATTERIES)
PATCH_SIZE = 30
FEATURE_CHUNK_WIDTH = 8


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    model_kind: str
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int


SPECS = [
    TransformerSpec(
        name="feature_tokens_h4_d64_l2",
        model_kind="feature_only",
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.12,
        learning_rate=7.0e-4,
        weight_decay=5.0e-4,
        epochs=18,
        batch_size=4096,
    ),
    TransformerSpec(
        name="pure_full_tokens_h4_d64_l2",
        model_kind="patch_plus_features",
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.15,
        learning_rate=5.0e-4,
        weight_decay=5.0e-4,
        epochs=10,
        batch_size=4096,
    ),
    TransformerSpec(
        name="hybrid_patch_full_h4_d64_l2",
        model_kind="hybrid",
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.15,
        learning_rate=5.0e-4,
        weight_decay=5.0e-4,
        epochs=10,
        batch_size=4096,
    ),
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    pct_err = abs_err / np.clip(np.abs(y_true), 1.0e-8, None)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(pct_err.mean() * 100.0),
        "within_2pct": float((pct_err <= 0.02).mean()),
        "within_5pct": float((pct_err <= 0.05).mean()),
        "within_10pct": float((pct_err <= 0.10).mean()),
    }


def standardize_target(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(train_y.mean(dtype=np.float64))
    std = float(train_y.std(dtype=np.float64))
    if std < 1.0e-8:
        std = 1.0
    return ((train_y - mean) / std).astype(np.float32), ((test_y - mean) / std).astype(np.float32), mean, std


def destandardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values.astype(np.float32) * std + mean


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


def safe_slope(delta_y: float, delta_t: float) -> float:
    return float(delta_y / max(delta_t, 1.0e-8))


def sequence_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    clean = values.astype(np.float64, copy=False)
    std = float(np.std(clean))
    centered = clean - float(np.mean(clean))
    if std < 1.0e-12:
        skew = 0.0
        kurt = 0.0
    else:
        z = centered / std
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4) - 3.0)
    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_std": std,
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
        f"{prefix}_q10": float(np.quantile(clean, 0.10)),
        f"{prefix}_q25": float(np.quantile(clean, 0.25)),
        f"{prefix}_median": float(np.quantile(clean, 0.50)),
        f"{prefix}_q75": float(np.quantile(clean, 0.75)),
        f"{prefix}_q90": float(np.quantile(clean, 0.90)),
        f"{prefix}_skew": skew,
        f"{prefix}_kurtosis": kurt,
    }


def event_features(
    battery: str,
    event_type: str,
    event_order: int,
    mat_step_index: int,
    date: str,
    label_charge_count: int,
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    rel_time: np.ndarray,
    abs_time: np.ndarray,
    checkpoint_counts: np.ndarray,
    capacities: np.ndarray,
    initial_capacity: float,
) -> dict[str, Any]:
    assigned_checkpoint, target_capacity, target_soh = assign_next_checkpoint(
        int(label_charge_count),
        checkpoint_counts,
        capacities,
        initial_capacity,
    )
    duration = float(rel_time[-1] - rel_time[0]) if len(rel_time) > 1 else 0.0
    dt = np.clip(np.diff(rel_time).astype(np.float64), 0.0, None)
    abs_current = np.abs(current.astype(np.float64, copy=False))
    if len(voltage) > 1:
        charge_ah = float(np.sum(abs_current[:-1] * dt) / 3600.0)
        energy_wh = float(np.sum(voltage[:-1] * abs_current[:-1] * dt) / 3600.0)
        dv = np.diff(voltage).astype(np.float64)
        dtemp = np.diff(temperature).astype(np.float64)
        valid_dt = dt > 1.0e-8
        dv_dt = np.divide(dv, np.where(valid_dt, dt, np.nan))
        dtemp_dt = np.divide(dtemp, np.where(valid_dt, dt, np.nan))
        dq = abs_current[:-1] * dt / 3600.0
        valid_dv = np.abs(dv) > 1.0e-6
        dq_dv = dq[valid_dv] / np.abs(dv[valid_dv]) if np.any(valid_dv) else np.array([0.0])
        dv_dq = np.abs(dv[dq > 1.0e-10]) / dq[dq > 1.0e-10] if np.any(dq > 1.0e-10) else np.array([0.0])
    else:
        charge_ah = 0.0
        energy_wh = 0.0
        dv_dt = np.array([0.0])
        dtemp_dt = np.array([0.0])
        dq_dv = np.array([0.0])
        dv_dq = np.array([0.0])

    time_in_38_40 = float(np.sum(np.diff(rel_time, prepend=rel_time[0])[(voltage >= 3.8) & (voltage < 4.0)]))
    time_in_40_41 = float(np.sum(np.diff(rel_time, prepend=rel_time[0])[(voltage >= 4.0) & (voltage < 4.1)]))
    cross_mask = (voltage >= 3.9) & (voltage <= 4.1)
    time_cross = float(rel_time[cross_mask][-1] - rel_time[cross_mask][0]) if np.sum(cross_mask) > 1 else 0.0
    power = voltage * abs_current
    features: dict[str, Any] = {
        "battery": battery,
        "event_type": event_type,
        "event_type_discharge": 1.0 if event_type == "discharge" else 0.0,
        "event_order": int(event_order),
        "mat_step_index": int(mat_step_index),
        "date": date,
        "label_charge_count": int(label_charge_count),
        "assigned_checkpoint_cycle": int(assigned_checkpoint),
        "capacity_ah": float(target_capacity),
        "soh_percent": float(target_soh),
        "initial_capacity_ah": float(initial_capacity),
        "sample_count": int(len(voltage)),
        "duration_s": duration,
        "time_start": float(abs_time[0]),
        "time_end": float(abs_time[-1]),
        "charge_throughput_ah": charge_ah,
        "energy_throughput_wh": energy_wh,
        "power_mean_w": float(np.mean(power)),
        "power_max_w": float(np.max(power)),
        "voltage_start": float(voltage[0]),
        "voltage_end": float(voltage[-1]),
        "voltage_delta": float(voltage[-1] - voltage[0]),
        "voltage_abs_delta": float(abs(voltage[-1] - voltage[0])),
        "voltage_slope_vps": safe_slope(float(voltage[-1] - voltage[0]), duration),
        "dv_dt_mean_vps": float(np.nanmean(dv_dt)),
        "dv_dt_median_vps": float(np.nanmedian(dv_dt)),
        "dv_dt_max_abs_vps": float(np.nanmax(np.abs(dv_dt))),
        "current_mean_a": float(np.mean(current)),
        "current_abs_mean_a": float(np.mean(abs_current)),
        "current_std_a": float(np.std(current)),
        "current_min_a": float(np.min(current)),
        "current_max_a": float(np.max(current)),
        "temperature_start_c": float(temperature[0]),
        "temperature_end_c": float(temperature[-1]),
        "temperature_delta_c": float(temperature[-1] - temperature[0]),
        "temperature_abs_delta_c": float(abs(temperature[-1] - temperature[0])),
        "temperature_mean_c": float(np.mean(temperature)),
        "temperature_std_c": float(np.std(temperature)),
        "dtemp_dt_mean_cps": float(np.nanmean(dtemp_dt)),
        "dtemp_dt_median_cps": float(np.nanmedian(dtemp_dt)),
        "dtemp_dt_max_abs_cps": float(np.nanmax(np.abs(dtemp_dt))),
        "time_in_vwin_3p8_4p0_s": time_in_38_40,
        "time_in_vwin_4p0_4p1_s": time_in_40_41,
        "time_cross_3p9_to_4p1_s": time_cross,
        "dq_dv_median_ahpv": float(np.median(dq_dv)),
        "dq_dv_max_ahpv": float(np.max(dq_dv)),
        "dq_dv_min_ahpv": float(np.min(dq_dv)),
        "dv_dq_median_vpah": float(np.median(dv_dq)),
        "dv_dq_max_vpah": float(np.max(dv_dq)),
        "dv_dq_min_vpah": float(np.min(dv_dq)),
    }
    features.update(sequence_stats(voltage, "voltage_pdf"))
    features.update(sequence_stats(temperature, "temperature_pdf"))
    return features


def build_patch_tokens(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    rel_time: np.ndarray,
    abs_time: np.ndarray,
    battery_time0: float,
) -> np.ndarray:
    raw_patch = build_patch_sequence(
        voltage.astype(np.float32),
        current.astype(np.float32),
        temperature.astype(np.float32),
        rel_time.astype(np.float32),
        abs_time.astype(np.float64),
        battery_time0,
    )
    v_scaled = minmax01(voltage)
    t_scaled = minmax01(temperature)
    c_scaled = maxabs01(current)
    r_scaled = minmax01(rel_time)
    rows: list[np.ndarray] = []
    token_count = math.ceil(len(voltage) / PATCH_SIZE)
    for patch_idx, start in enumerate(range(0, len(voltage), PATCH_SIZE)):
        end = min(start + PATCH_SIZE, len(voltage))
        pv = v_scaled[start:end]
        pt = t_scaled[start:end]
        pc = c_scaled[start:end]
        pr = r_scaled[start:end]
        rows.append(
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
    return np.concatenate([raw_patch, np.stack(rows, axis=0)], axis=1).astype(np.float32, copy=False)


def build_or_load_event_table_and_patches() -> tuple[pd.DataFrame, list[np.ndarray]]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    features_path = OUTDIR / "charge_discharge_full_event_features.csv"
    patches_path = OUTDIR / "charge_discharge_patch_tokens.pkl"
    if features_path.exists() and patches_path.exists():
        features = pd.read_csv(features_path)
        with patches_path.open("rb") as f:
            patches = pickle.load(f)
        return features, patches

    all_rows: list[dict[str, Any]] = []
    all_patches: list[np.ndarray] = []
    for battery in BATTERIES:
        checkpoint_counts, capacities, initial_capacity = load_reference_label_table(battery)
        battery_times: list[float] = []
        for event_type in ("charge", "discharge"):
            df = pd.read_pickle(SEGMENT_DIR / f"{battery}_{event_type}_random_walk_segments.pkl")
            battery_times.append(float(df["time"].min()))
        battery_time0 = min(battery_times)

        for event_type in ("charge", "discharge"):
            path = SEGMENT_DIR / f"{battery}_{event_type}_random_walk_segments.pkl"
            df = pd.read_pickle(path).sort_values(["event_order", "sample_index"]).reset_index(drop=True)
            event_order = df["event_order"].to_numpy(dtype=np.int64)
            unique_orders, starts = np.unique(event_order, return_index=True)
            ends = np.r_[starts[1:], len(df)]
            voltage_all = df["voltage"].to_numpy(dtype=np.float64)
            current_all = df["current"].to_numpy(dtype=np.float64)
            temperature_all = df["temperature"].to_numpy(dtype=np.float64)
            rel_time_all = df["relativeTime"].to_numpy(dtype=np.float64)
            abs_time_all = df["time"].to_numpy(dtype=np.float64)
            mat_step_all = df["mat_step_index"].to_numpy(dtype=np.int64)
            label_count_all = df["label_charge_count"].to_numpy(dtype=np.int64)
            date_all = df["date"].to_numpy(dtype=object)
            for idx, (order, start, end) in enumerate(zip(unique_orders, starts, ends), start=1):
                v = voltage_all[start:end]
                c = current_all[start:end]
                t = temperature_all[start:end]
                rt = rel_time_all[start:end]
                at = abs_time_all[start:end]
                row = event_features(
                    battery=battery,
                    event_type=event_type,
                    event_order=int(order),
                    mat_step_index=int(mat_step_all[start]),
                    date=str(date_all[start]),
                    label_charge_count=int(label_count_all[start]),
                    voltage=v,
                    current=c,
                    temperature=t,
                    rel_time=rt,
                    abs_time=at,
                    checkpoint_counts=checkpoint_counts,
                    capacities=capacities,
                    initial_capacity=initial_capacity,
                )
                all_rows.append(row)
                all_patches.append(build_patch_tokens(v, c, t, rt, at, battery_time0))
                if idx % 5000 == 0:
                    print(f"{battery} {event_type}: processed {idx}/{len(unique_orders)} events", flush=True)
    features = pd.DataFrame(all_rows)
    features.to_csv(features_path, index=False)
    with patches_path.open("wb") as f:
        pickle.dump(all_patches, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = (
        features.groupby(["battery", "event_type"], as_index=False)
        .agg(events=("event_order", "count"), rows=("sample_count", "sum"), checkpoints=("assigned_checkpoint_cycle", "nunique"))
    )
    summary.to_csv(OUTDIR / "event_summary.csv", index=False)
    return features, all_patches


def standardize_matrix(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.nan_to_num(train_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(test_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def standardize_patches(
    train_patches: list[np.ndarray],
    test_patches: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    stacked = np.concatenate([np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in train_patches], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    train_out = [((np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std).astype(np.float32) for x in train_patches]
    test_out = [((np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std).astype(np.float32) for x in test_patches]
    return train_out, test_out, mean, std


class FullCycleDataset(Dataset):
    def __init__(
        self,
        feature_x: np.ndarray,
        target_y: np.ndarray,
        weights: np.ndarray,
        meta: pd.DataFrame,
        patch_sequences: list[np.ndarray] | None = None,
    ):
        self.feature_x = feature_x.astype(np.float32, copy=False)
        self.target_y = target_y.astype(np.float32, copy=False)
        self.weights = weights.astype(np.float32, copy=False)
        self.meta = meta.reset_index(drop=True)
        self.patch_sequences = patch_sequences

    def __len__(self) -> int:
        return int(len(self.target_y))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.iloc[idx]
        out = {
            "feature_x": self.feature_x[idx],
            "target": float(self.target_y[idx]),
            "weight": float(self.weights[idx]),
            "battery": str(row["battery"]),
            "event_type": str(row["event_type"]),
            "event_order": int(row["event_order"]),
            "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
            "sample_id": f"{row['battery']}_{row['event_type']}_{int(row['event_order'])}",
        }
        if self.patch_sequences is not None:
            out["patch_sequence"] = self.patch_sequences[idx]
        return out


def collate_full_batch(batch: list[dict[str, Any]]) -> tuple[Any, ...]:
    feature_x = torch.tensor(np.stack([item["feature_x"] for item in batch]), dtype=torch.float32)
    target = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
    weight = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)
    batteries = [item["battery"] for item in batch]
    event_types = [item["event_type"] for item in batch]
    event_orders = [int(item["event_order"]) for item in batch]
    checkpoints = [int(item["assigned_checkpoint_cycle"]) for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    if "patch_sequence" not in batch[0]:
        return feature_x, target, weight, batteries, event_types, event_orders, checkpoints, sample_ids

    max_len = max(len(item["patch_sequence"]) for item in batch)
    patch_dim = batch[0]["patch_sequence"].shape[1]
    patch_x = torch.zeros((len(batch), max_len, patch_dim), dtype=torch.float32)
    patch_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for idx, item in enumerate(batch):
        seq = torch.from_numpy(item["patch_sequence"]).float()
        patch_x[idx, : seq.shape[0]] = seq
        patch_mask[idx, : seq.shape[0]] = True
    return patch_x, patch_mask, feature_x, target, weight, batteries, event_types, event_orders, checkpoints, sample_ids


class FeatureTokenTransformer(nn.Module):
    def __init__(self, feature_count: int, spec: TransformerSpec):
        super().__init__()
        self.feature_count = feature_count
        self.chunk_count = int(math.ceil(feature_count / FEATURE_CHUNK_WIDTH))
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.feature_proj = nn.Linear(FEATURE_CHUNK_WIDTH, spec.d_model)
        self.feature_id_embedding = nn.Embedding(self.chunk_count, spec.d_model)
        self.input_norm = nn.LayerNorm(spec.d_model)
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

    def feature_tokens(self, feature_x: torch.Tensor) -> torch.Tensor:
        batch_size = feature_x.size(0)
        pad = self.chunk_count * FEATURE_CHUNK_WIDTH - self.feature_count
        if pad:
            feature_x = nn.functional.pad(feature_x, (0, pad))
        chunks = feature_x.view(batch_size, self.chunk_count, FEATURE_CHUNK_WIDTH)
        ids = torch.arange(self.chunk_count, device=feature_x.device)
        return self.feature_proj(chunks) + self.feature_id_embedding(ids).unsqueeze(0)

    def forward(self, feature_x: torch.Tensor) -> torch.Tensor:
        batch_size = feature_x.size(0)
        tokens = self.feature_tokens(feature_x)
        cls = self.cls.expand(batch_size, -1, -1)
        encoded = self.encoder(self.input_norm(torch.cat([cls, tokens], dim=1)))
        return self.head(encoded[:, 0]).squeeze(-1)


class PatchFeatureTokenTransformer(FeatureTokenTransformer):
    def __init__(self, patch_dim: int, feature_count: int, spec: TransformerSpec):
        super().__init__(feature_count, spec)
        self.patch_proj = nn.Linear(patch_dim, spec.d_model)
        self.type_embedding = nn.Embedding(3, spec.d_model)

    def forward(self, patch_x: torch.Tensor, patch_mask: torch.Tensor, feature_x: torch.Tensor) -> torch.Tensor:
        batch_size = patch_x.size(0)
        patch_tokens = self.patch_proj(patch_x)
        feature_tokens = self.feature_tokens(feature_x)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, patch_tokens, feature_tokens], dim=1)
        cls_type = torch.zeros((batch_size, 1), dtype=torch.long, device=feature_x.device)
        patch_type = torch.ones((batch_size, patch_tokens.size(1)), dtype=torch.long, device=feature_x.device)
        feature_type = torch.full((batch_size, feature_tokens.size(1)), 2, dtype=torch.long, device=feature_x.device)
        type_ids = torch.cat([cls_type, patch_type, feature_type], dim=1)
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=feature_x.device)
        feature_mask = torch.ones((batch_size, feature_tokens.size(1)), dtype=torch.bool, device=feature_x.device)
        mask = torch.cat([cls_mask, patch_mask, feature_mask], dim=1)
        encoded = self.encoder(self.input_norm(tokens + self.type_embedding(type_ids)), src_key_padding_mask=~mask)
        return self.head(encoded[:, 0]).squeeze(-1)


class HybridPatchFullTransformer(nn.Module):
    def __init__(self, patch_dim: int, feature_count: int, spec: TransformerSpec):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, spec.d_model))
        self.patch_proj = nn.Linear(patch_dim, spec.d_model)
        self.patch_norm = nn.LayerNorm(spec.d_model)
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
        self.full_branch = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_model, spec.d_model),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(spec.d_model * 2),
            nn.Linear(spec.d_model * 2, spec.d_model),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.d_model, 1),
        )

    def forward(self, patch_x: torch.Tensor, patch_mask: torch.Tensor, feature_x: torch.Tensor) -> torch.Tensor:
        batch_size = patch_x.size(0)
        patch_tokens = self.patch_norm(self.patch_proj(patch_x))
        cls = self.cls.expand(batch_size, -1, -1)
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=patch_x.device)
        mask = torch.cat([cls_mask, patch_mask], dim=1)
        encoded = self.encoder(torch.cat([cls, patch_tokens], dim=1), src_key_padding_mask=~mask)
        full = self.full_branch(feature_x)
        return self.head(torch.cat([encoded[:, 0], full], dim=1)).squeeze(-1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    model_kind: str,
) -> tuple[float, np.ndarray, np.ndarray, list[str], list[str], list[int], list[int], list[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batteries: list[str] = []
    event_types: list[str] = []
    event_orders: list[int] = []
    checkpoints: list[int] = []
    sample_ids: list[str] = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if model_kind == "feature_only":
                feature_x, y, w, b, et, eo, cp, sid = batch
                feature_x = feature_x.to(device)
                y = y.to(device)
                w = w.to(device)
                out = model(feature_x)
            else:
                patch_x, patch_mask, feature_x, y, w, b, et, eo, cp, sid = batch
                patch_x = patch_x.to(device)
                patch_mask = patch_mask.to(device)
                feature_x = feature_x.to(device)
                y = y.to(device)
                w = w.to(device)
                out = model(patch_x, patch_mask, feature_x)
            per_sample = (out - y) ** 2
            loss = (per_sample * w).sum() / torch.clamp(w.sum(), min=1.0e-8)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = int(y.size(0))
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            preds.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            batteries.extend(b)
            event_types.extend(et)
            event_orders.extend([int(x) for x in eo])
            checkpoints.extend([int(x) for x in cp])
            sample_ids.extend(sid)
    return total_loss / max(total_count, 1), np.concatenate(preds), np.concatenate(targets), batteries, event_types, event_orders, checkpoints, sample_ids


def prediction_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batteries: list[str],
    event_types: list[str],
    event_orders: list[int],
    checkpoints: list[int],
    sample_ids: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "battery": batteries,
            "event_type": event_types,
            "event_order": event_orders,
            "assigned_checkpoint_cycle": checkpoints,
            "actual_soh_percent": y_true,
            "predicted_soh_percent": y_pred,
            "error_pp": y_pred - y_true,
            "absolute_error_pp": np.abs(y_pred - y_true),
        }
    )


def aggregate_checkpoints(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        pred_df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False)
        .agg(
            actual_soh_percent=("actual_soh_percent", "mean"),
            predicted_soh_percent=("predicted_soh_percent", "mean"),
            event_count=("sample_id", "count"),
            charge_events=("event_type", lambda x: int((x == "charge").sum())),
            discharge_events=("event_type", lambda x: int((x == "discharge").sum())),
        )
        .sort_values(["battery", "assigned_checkpoint_cycle"])
        .reset_index(drop=True)
    )
    out["error_pp"] = out["predicted_soh_percent"] - out["actual_soh_percent"]
    out["absolute_error_pp"] = out["error_pp"].abs()
    return out


def save_prediction_plots(outdir: Path, name: str, history: pd.DataFrame, pred_df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["test_loss"], label="test")
    axes[0].set_title("Weighted MSE loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(history["epoch"], history["train_r2"], label="train")
    axes[1].plot(history["epoch"], history["test_r2"], label="test")
    axes[1].set_title("R2")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(history["epoch"], history["train_mae"], label="train")
    axes[2].plot(history["epoch"], history["test_mae"], label="test")
    axes[2].set_title("MAE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(outdir / "training_dashboard.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = pred_df["event_type"].map({"charge": "#2a9d8f", "discharge": "#e76f51"}).to_numpy()
    axes[0, 0].scatter(pred_df["actual_soh_percent"], pred_df["predicted_soh_percent"], s=7, alpha=0.25, c=colors)
    lo = min(pred_df["actual_soh_percent"].min(), pred_df["predicted_soh_percent"].min())
    hi = max(pred_df["actual_soh_percent"].max(), pred_df["predicted_soh_percent"].max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--")
    axes[0, 0].set_title("Actual vs predicted")
    axes[0, 0].set_xlabel("Actual SOH (%)")
    axes[0, 0].set_ylabel("Predicted SOH (%)")
    axes[0, 0].grid(True, alpha=0.25)
    for event_type, color in [("charge", "#2a9d8f"), ("discharge", "#e76f51")]:
        sub = pred_df[pred_df["event_type"].eq(event_type)]
        axes[0, 1].scatter(sub["event_order"], sub["actual_soh_percent"], s=5, alpha=0.15, color="black")
        axes[0, 1].scatter(sub["event_order"], sub["predicted_soh_percent"], s=5, alpha=0.15, color=color, label=event_type)
    axes[0, 1].set_title("RW11 SOH over event order")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["actual_soh_percent"], marker="o", label="actual")
    axes[1, 0].plot(checkpoint_df["assigned_checkpoint_cycle"], checkpoint_df["predicted_soh_percent"], marker="o", label="predicted")
    axes[1, 0].set_title("Checkpoint-aggregated prediction")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 1].hist(pred_df["error_pp"], bins=60, color="#457b9d")
    axes[1, 1].axvline(0, color="black", linestyle="--")
    axes[1, 1].set_title("Residuals")
    axes[1, 1].grid(True, alpha=0.25)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(outdir / "prediction_report.png", dpi=180)
    plt.close(fig)


def train_transformer(
    spec: TransformerSpec,
    train_feature_x: np.ndarray,
    test_feature_x: np.ndarray,
    train_patch: list[np.ndarray],
    test_patch: list[np.ndarray],
    train_y_norm: np.ndarray,
    test_y_norm: np.ndarray,
    train_w: np.ndarray,
    test_w: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> dict[str, Any]:
    outdir = OUTDIR / spec.name
    outdir.mkdir(parents=True, exist_ok=True)
    needs_patch = spec.model_kind in {"patch_plus_features", "hybrid"}
    train_ds = FullCycleDataset(train_feature_x, train_y_norm, train_w, train_meta, train_patch if needs_patch else None)
    test_ds = FullCycleDataset(test_feature_x, test_y_norm, test_w, test_meta, test_patch if needs_patch else None)
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, num_workers=0, collate_fn=collate_full_batch)
    test_loader = DataLoader(test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0, collate_fn=collate_full_batch)
    if spec.model_kind == "feature_only":
        model: nn.Module = FeatureTokenTransformer(train_feature_x.shape[1], spec)
    elif spec.model_kind == "patch_plus_features":
        model = PatchFeatureTokenTransformer(train_patch[0].shape[1], train_feature_x.shape[1], spec)
    elif spec.model_kind == "hybrid":
        model = HybridPatchFullTransformer(train_patch[0].shape[1], train_feature_x.shape[1], spec)
    else:
        raise ValueError(f"Unknown model kind: {spec.model_kind}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    history_rows: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_r2 = -np.inf
    for epoch in range(1, spec.epochs + 1):
        train_loss, train_pred_norm, train_true_norm, *_ = run_epoch(model, train_loader, optimizer, device, spec.model_kind)
        test_loss, test_pred_norm, test_true_norm, *_ = run_epoch(model, test_loader, None, device, spec.model_kind)
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
        if test_metrics["r2"] > best_r2:
            best_r2 = test_metrics["r2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"{spec.name} epoch {epoch}/{spec.epochs}: "
            f"train_r2={train_metrics['r2']:.4f} test_r2={test_metrics['r2']:.4f} test_mae={test_metrics['mae']:.3f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = run_epoch(model, train_loader, None, device, spec.model_kind)
    test_eval = run_epoch(model, test_loader, None, device, spec.model_kind)
    _, train_pred_norm, train_true_norm, train_b, train_et, train_eo, train_cp, train_sid = train_eval
    _, test_pred_norm, test_true_norm, test_b, test_et, test_eo, test_cp, test_sid = test_eval
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)
    train_pred_df = prediction_frame(train_true, train_pred, train_b, train_et, train_eo, train_cp, train_sid)
    test_pred_df = prediction_frame(test_true, test_pred, test_b, test_et, test_eo, test_cp, test_sid)
    checkpoint_df = aggregate_checkpoints(test_pred_df)
    history = pd.DataFrame(history_rows)
    metrics = {
        "spec": asdict(spec),
        "parameter_count": parameter_count(model),
        "train_sample_count": int(len(train_pred_df)),
        "test_sample_count": int(len(test_pred_df)),
        "feature_count": int(train_feature_x.shape[1]),
        "feature_token_count": int(math.ceil(train_feature_x.shape[1] / FEATURE_CHUNK_WIDTH)),
        "patch_token_dim": int(train_patch[0].shape[1]) if needs_patch else None,
        "max_patch_tokens": int(max(len(x) for x in train_patch + test_patch)) if needs_patch else None,
        "best_epoch_by_test_r2": int(history.loc[history["test_r2"].idxmax(), "epoch"]),
        "train_metrics_event_level": regression_metrics(train_true, train_pred),
        "test_metrics_event_level": regression_metrics(test_true, test_pred),
        "test_metrics_checkpoint_aggregated": regression_metrics(
            checkpoint_df["actual_soh_percent"].to_numpy(),
            checkpoint_df["predicted_soh_percent"].to_numpy(),
        ),
    }
    history.to_csv(outdir / "training_history.csv", index=False)
    train_pred_df.to_csv(outdir / "train_predictions.csv", index=False)
    test_pred_df.to_csv(outdir / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(outdir / "test_checkpoint_predictions.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "spec": asdict(spec), "metrics": metrics}, outdir / "model.pt")
    save_prediction_plots(outdir, spec.name, history, test_pred_df, checkpoint_df)
    return metrics


def run_lightgbm(
    train_feature_x: np.ndarray,
    test_feature_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    train_w: np.ndarray,
    test_meta: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    outdir = OUTDIR / "lightgbm_charge_discharge"
    outdir.mkdir(parents=True, exist_ok=True)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.75,
        colsample_bytree=0.60,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_feature_x, train_y, sample_weight=train_w)
    pred = model.predict(test_feature_x)
    pred_df = pd.DataFrame(
        {
            "sample_id": [f"{r.battery}_{r.event_type}_{int(r.event_order)}" for r in test_meta.itertuples(index=False)],
            "battery": test_meta["battery"].to_numpy(),
            "event_type": test_meta["event_type"].to_numpy(),
            "event_order": test_meta["event_order"].to_numpy(dtype=np.int64),
            "assigned_checkpoint_cycle": test_meta["assigned_checkpoint_cycle"].to_numpy(dtype=np.int64),
            "actual_soh_percent": test_y,
            "predicted_soh_percent": pred,
            "error_pp": pred - test_y,
            "absolute_error_pp": np.abs(pred - test_y),
        }
    )
    checkpoint_df = aggregate_checkpoints(pred_df)
    metrics = {
        "spec": {"name": "lightgbm_charge_discharge_full_event_features"},
        "train_sample_count": int(len(train_y)),
        "test_sample_count": int(len(test_y)),
        "feature_count": int(train_feature_x.shape[1]),
        "test_metrics_event_level": regression_metrics(test_y, pred),
        "test_metrics_checkpoint_aggregated": regression_metrics(
            checkpoint_df["actual_soh_percent"].to_numpy(),
            checkpoint_df["predicted_soh_percent"].to_numpy(),
        ),
    }
    pred_df.to_csv(outdir / "test_predictions.csv", index=False)
    checkpoint_df.to_csv(outdir / "test_checkpoint_predictions.csv", index=False)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(outdir / "feature_importance.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(test_y, pred, s=7, alpha=0.25, c=pred_df["event_type"].map({"charge": "#2a9d8f", "discharge": "#e76f51"}))
    lo = min(float(np.min(test_y)), float(np.min(pred)))
    hi = max(float(np.max(test_y)), float(np.max(pred)))
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set_title("LightGBM actual vs predicted")
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")
    axes[0].grid(True, alpha=0.25)
    top = importance.head(15).iloc[::-1]
    axes[1].barh(top["feature"], top["importance_gain"], color="#457b9d")
    axes[1].set_title("Top LightGBM features")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "prediction_and_importance.png", dpi=180)
    plt.close(fig)
    return metrics


def read_existing_patch_only_row() -> dict[str, Any]:
    path = ROOT / "charge_discharge_rw11_tuning" / "h4_d80_drop020_wd5e4" / "metrics.json"
    if not path.exists():
        path = ROOT / "charge_discharge_weighted_transformer" / "rw9_rw10_to_rw11" / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    spec = metrics["spec"]
    return {
        "model": "patch_only_transformer_charge_discharge",
        "model_family": "transformer",
        "input_scope": "charge+discharge",
        "test_samples": int(metrics["test_sample_count"]) if "test_sample_count" in metrics else 46490,
        "test_r2": float(metrics["test_metrics_event_level"]["r2"]),
        "test_mae": float(metrics["test_metrics_event_level"]["mae"]),
        "test_rmse": float(metrics["test_metrics_event_level"]["rmse"]),
        "checkpoint_r2": float(metrics["test_metrics_checkpoint_aggregated"]["r2"]),
        "checkpoint_mae": float(metrics["test_metrics_checkpoint_aggregated"]["mae"]),
        "d_model": spec.get("d_model"),
        "nhead": spec.get("nhead"),
        "num_layers": spec.get("num_layers"),
        "dropout": spec.get("dropout"),
        "learning_rate": spec.get("learning_rate"),
        "epochs": spec.get("epochs"),
        "output_dir": str(path.parent),
    }


def save_suite_comparison(rows: list[dict[str, Any]]) -> None:
    summary = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    summary.to_csv(OUTDIR / "model_suite_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = summary["model"].tolist()
    x = np.arange(len(labels))
    for ax, col, title, color in zip(
        axes,
        ["test_r2", "test_mae", "checkpoint_r2"],
        ["RW11 event-level R2", "RW11 event-level MAE", "RW11 checkpoint R2"],
        ["#2a9d8f", "#457b9d", "#f4a261"],
    ):
        ax.bar(x, summary[col], color=color)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "model_suite_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(8)
    features, patches = build_or_load_event_table_and_patches()
    features = add_block_weights(features.rename(columns={"label_charge_count": "charge_cycle_for_label"}))
    features = features.rename(columns={"charge_cycle_for_label": "label_charge_count"})
    train_mask = features["battery"].isin(TRAIN_BATTERIES).to_numpy()
    test_mask = features["battery"].isin(TEST_BATTERIES).to_numpy()
    train_meta = features.loc[train_mask].reset_index(drop=True)
    test_meta = features.loc[test_mask].reset_index(drop=True)
    train_patches_raw = [patches[i] for i in np.flatnonzero(train_mask)]
    test_patches_raw = [patches[i] for i in np.flatnonzero(test_mask)]

    exclude = {
        "battery",
        "event_type",
        "event_order",
        "mat_step_index",
        "date",
        "assigned_checkpoint_cycle",
        "capacity_ah",
        "soh_percent",
        "initial_capacity_ah",
        "block_size",
        "sample_weight",
    }
    feature_names = [
        col
        for col in features.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(features[col])
    ]
    train_feature_raw = train_meta[feature_names].to_numpy(dtype=np.float32)
    test_feature_raw = test_meta[feature_names].to_numpy(dtype=np.float32)
    train_feature_x, test_feature_x, feature_mean, feature_std = standardize_matrix(train_feature_raw, test_feature_raw)
    train_patches, test_patches, patch_mean, patch_std = standardize_patches(train_patches_raw, test_patches_raw)

    train_y = train_meta["soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    test_w = test_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, test_y_norm, y_mean, y_std = standardize_target(train_y, test_y)

    pd.DataFrame({"feature": feature_names, "mean": feature_mean, "std": feature_std}).to_csv(
        OUTDIR / "feature_scaler.csv",
        index=False,
    )
    pd.DataFrame({"patch_feature_index": np.arange(len(patch_mean)), "mean": patch_mean, "std": patch_std}).to_csv(
        OUTDIR / "patch_scaler.csv",
        index=False,
    )
    (OUTDIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = [read_existing_patch_only_row()]
    lgb_metrics = run_lightgbm(train_feature_raw, test_feature_raw, train_y, test_y, train_w, test_meta, feature_names)
    rows.append(
        {
            "model": "lightgbm_charge_discharge",
            "model_family": "lightgbm",
            "input_scope": "charge+discharge",
            "test_samples": lgb_metrics["test_sample_count"],
            "test_r2": lgb_metrics["test_metrics_event_level"]["r2"],
            "test_mae": lgb_metrics["test_metrics_event_level"]["mae"],
            "test_rmse": lgb_metrics["test_metrics_event_level"]["rmse"],
            "checkpoint_r2": lgb_metrics["test_metrics_checkpoint_aggregated"]["r2"],
            "checkpoint_mae": lgb_metrics["test_metrics_checkpoint_aggregated"]["mae"],
            "d_model": None,
            "nhead": None,
            "num_layers": None,
            "dropout": None,
            "learning_rate": 0.05,
            "epochs": 1000,
            "output_dir": str(OUTDIR / "lightgbm_charge_discharge"),
        }
    )

    device = torch.device("cpu")
    for spec in SPECS:
        set_seed()
        metrics = train_transformer(
            spec,
            train_feature_x,
            test_feature_x,
            train_patches,
            test_patches,
            train_y_norm,
            test_y_norm,
            train_w,
            test_w,
            train_meta,
            test_meta,
            y_mean,
            y_std,
            device,
        )
        rows.append(
            {
                "model": spec.name,
                "model_family": "transformer",
                "input_scope": "charge+discharge",
                "test_samples": metrics["test_sample_count"],
                "test_r2": metrics["test_metrics_event_level"]["r2"],
                "test_mae": metrics["test_metrics_event_level"]["mae"],
                "test_rmse": metrics["test_metrics_event_level"]["rmse"],
                "checkpoint_r2": metrics["test_metrics_checkpoint_aggregated"]["r2"],
                "checkpoint_mae": metrics["test_metrics_checkpoint_aggregated"]["mae"],
                "d_model": spec.d_model,
                "nhead": spec.nhead,
                "num_layers": spec.num_layers,
                "dropout": spec.dropout,
                "learning_rate": spec.learning_rate,
                "epochs": spec.epochs,
                "output_dir": str(OUTDIR / spec.name),
            }
        )

    save_suite_comparison(rows)
    print(pd.DataFrame(rows).sort_values("test_r2", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
