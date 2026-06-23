from __future__ import annotations

import json
import math
import pickle
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
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_charge_discharge_weighted_transformers import assign_next_checkpoint, load_reference_label_table


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
SEGMENT_DIR = ROOT / "charge_discharge_weighted_transformer" / "extracted_segments"
OUT = ROOT / "charge_discharge_top20_event_transformer_xai_ppt"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)
BATTERIES = (*TRAIN_BATTERIES, *TEST_BATTERIES)
WINDOW = 32
RANDOM_SEED = 42

NAVY = RGBColor(19, 43, 74)
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(15, 23, 42)
SLATE = RGBColor(71, 85, 105)
LIGHT_BLUE = RGBColor(232, 244, 252)
LIGHT_GRAY = RGBColor(248, 250, 252)
ORANGE = RGBColor(234, 88, 12)
TEAL = RGBColor(20, 135, 120)

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
)


@dataclass
class Config:
    window: int = WINDOW
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn: int = 128
    dropout: float = 0.10
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    epochs: int = 12
    batch_size: int = 4096
    val_frac: float = 0.10
    clip_quantile: float = 0.01
    seed: int = RANDOM_SEED


CFG = Config()


FEATURES: list[str] = [
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


FEATURE_DEFS: dict[str, dict[str, str]] = {
    "event_type_discharge": {
        "label": "Event direction indicator",
        "source": "Step comment/current sign",
        "equation": "z_D=1 if discharge, otherwise z_D=0",
        "unit": "unitless",
        "meaning": "Separates discharge events from charge events so current magnitude is not ambiguous.",
        "change": "Added for charge+discharge model",
    },
    "duration_s": {
        "label": "Event duration",
        "source": "relTime",
        "equation": "T=t_N-t_1",
        "unit": "s",
        "meaning": "How long the random-walk event lasted.",
        "change": "Kept",
    },
    "current_mean_a": {
        "label": "Signed mean current",
        "source": "Current",
        "equation": "I_{mean}=N^{-1}\\sum_{k=1}^{N}I_k",
        "unit": "A",
        "meaning": "Preserves direction: negative is charge, positive is discharge.",
        "change": "Added for charge+discharge model",
    },
    "current_abs_mean_a": {
        "label": "Mean absolute current",
        "source": "Current",
        "equation": "I_{abs,mean}=N^{-1}\\sum_{k=1}^{N}|I_k|",
        "unit": "A",
        "meaning": "Average loading magnitude during the event.",
        "change": "Meaning updated: magnitude only, direction handled separately",
    },
    "current_throughput_signed_ah": {
        "label": "Signed current throughput",
        "source": "Current + relTime",
        "equation": "Q_{signed}=3600^{-1}\\int I(t)dt",
        "unit": "Ah",
        "meaning": "Net current-time exposure; sign distinguishes discharge from charge.",
        "change": "Added for charge+discharge model",
    },
    "throughput_magnitude_ah": {
        "label": "Current throughput magnitude",
        "source": "Current + relTime",
        "equation": "Q_{mag}=3600^{-1}\\int |I(t)|dt",
        "unit": "Ah",
        "meaning": "Total electrical charge moved regardless of direction.",
        "change": "Renamed from charge throughput",
    },
    "energy_signed_wh": {
        "label": "Signed energy throughput",
        "source": "Voltage + Current + relTime",
        "equation": "E_{signed}=3600^{-1}\\int V(t)I(t)dt",
        "unit": "Wh",
        "meaning": "Net energy direction; positive for discharge and negative for charge in this dataset.",
        "change": "Added for charge+discharge model",
    },
    "energy_magnitude_wh": {
        "label": "Energy throughput magnitude",
        "source": "Voltage + Current + relTime",
        "equation": "E_{mag}=3600^{-1}\\int V(t)|I(t)|dt",
        "unit": "Wh",
        "meaning": "Electrical energy moved regardless of direction.",
        "change": "Renamed from energy throughput",
    },
    "power_mean_signed_w": {
        "label": "Signed mean power",
        "source": "Voltage + Current",
        "equation": "P_{mean}=N^{-1}\\sum_{k=1}^{N}V_kI_k",
        "unit": "W",
        "meaning": "Average power with direction retained.",
        "change": "Replaces charge-only mean charging power",
    },
    "power_mean_magnitude_w": {
        "label": "Mean power magnitude",
        "source": "Voltage + Current",
        "equation": "P_{mag,mean}=N^{-1}\\sum_{k=1}^{N}|V_kI_k|",
        "unit": "W",
        "meaning": "Average electrical loading severity regardless of direction.",
        "change": "Renamed from mean charging power",
    },
    "voltage_start": {
        "label": "Starting voltage",
        "source": "Voltage",
        "equation": "V_{start}=V_1",
        "unit": "V",
        "meaning": "Voltage state at the beginning of the event.",
        "change": "Added",
    },
    "voltage_end": {
        "label": "Ending voltage",
        "source": "Voltage",
        "equation": "V_{end}=V_N",
        "unit": "V",
        "meaning": "Voltage state at the end of the event.",
        "change": "Kept",
    },
    "voltage_delta_signed": {
        "label": "Signed voltage change",
        "source": "Voltage",
        "equation": "\\Delta V=V_N-V_1",
        "unit": "V",
        "meaning": "Voltage rises during charge and usually falls during discharge.",
        "change": "Renamed from voltage rise",
    },
    "voltage_pdf_std": {
        "label": "Voltage distribution spread",
        "source": "Voltage",
        "equation": "\\sigma_V=\\sqrt{(N-1)^{-1}\\sum_{k=1}^{N}(V_k-\\bar{V})^2}",
        "unit": "V",
        "meaning": "Width of the voltage region covered by the event.",
        "change": "Kept",
    },
    "voltage_pdf_min": {
        "label": "Minimum voltage",
        "source": "Voltage",
        "equation": "V_{min}=\\min_{1\\leq k\\leq N}V_k",
        "unit": "V",
        "meaning": "Lowest voltage inside the event.",
        "change": "Kept",
    },
    "voltage_pdf_q75": {
        "label": "75th percentile voltage",
        "source": "Voltage",
        "equation": "V_{75}=P_{75}(\\{V_1,V_2,\\ldots,V_N\\})",
        "unit": "V",
        "meaning": "Upper-quartile voltage inside the event.",
        "change": "Kept",
    },
    "voltage_pdf_q90": {
        "label": "90th percentile voltage",
        "source": "Voltage",
        "equation": "V_{90}=P_{90}(\\{V_1,V_2,\\ldots,V_N\\})",
        "unit": "V",
        "meaning": "High-end voltage value inside the event.",
        "change": "Kept",
    },
    "voltage_pdf_max": {
        "label": "Maximum voltage",
        "source": "Voltage",
        "equation": "V_{max}=\\max_{1\\leq k\\leq N}V_k",
        "unit": "V",
        "meaning": "Highest voltage inside the event.",
        "change": "Kept",
    },
    "voltage_pdf_skew": {
        "label": "Voltage distribution skewness",
        "source": "Voltage",
        "equation": "\\gamma_V=N^{-1}\\sum_{k=1}^{N}\\left((V_k-\\bar{V})/\\sigma_V\\right)^3",
        "unit": "unitless",
        "meaning": "Asymmetry of the voltage values inside the event.",
        "change": "Kept",
    },
    "dq_dv_max_ahpv": {
        "label": "Maximum local charge-change per voltage-change, dQ/dV",
        "source": "Voltage + Current + relTime",
        "equation": "(dQ/dV)_{max}=\\max_k(\\Delta Q_k/|\\Delta V_k|)",
        "unit": "Ah/V",
        "meaning": "Strongest local charge movement per voltage change; a shape feature, not the SOH label.",
        "change": "Kept with direction-safe formulation",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


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
    }


def sequence_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = values.astype(np.float64, copy=False)
    std = float(np.std(values))
    if std < 1e-12:
        skew = 0.0
    else:
        skew = float(np.mean(((values - float(np.mean(values))) / std) ** 3))
    return {
        f"{prefix}_std": std,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_skew": skew,
    }


def trapz(values: np.ndarray, rel_time: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.trapz(values.astype(np.float64), rel_time.astype(np.float64)))


def build_one_event_features(group: pd.DataFrame, checkpoint_counts: np.ndarray, capacities: np.ndarray, initial_capacity: float) -> dict[str, Any]:
    group = group.sort_values("sample_index")
    voltage = group["voltage"].to_numpy(dtype=np.float64)
    current = group["current"].to_numpy(dtype=np.float64)
    rel_time = group["relativeTime"].to_numpy(dtype=np.float64)
    label_count = int(group["label_charge_count"].iloc[0])
    checkpoint, capacity, soh = assign_next_checkpoint(label_count, checkpoint_counts, capacities, initial_capacity)
    event_type = str(group["event_type"].iloc[0])
    duration = float(rel_time[-1] - rel_time[0]) if len(rel_time) > 1 else 0.0
    current_throughput_signed = trapz(current, rel_time) / 3600.0
    throughput_magnitude = trapz(np.abs(current), rel_time) / 3600.0
    power_signed = voltage * current
    power_mag = np.abs(power_signed)
    energy_signed = trapz(power_signed, rel_time) / 3600.0
    energy_mag = trapz(power_mag, rel_time) / 3600.0
    if len(voltage) > 1:
        dt = np.clip(np.diff(rel_time), 0.0, None)
        dv = np.diff(voltage)
        dq = np.abs(current[:-1]) * dt / 3600.0
        valid = np.abs(dv) > 1e-6
        dq_dv = dq[valid] / np.abs(dv[valid]) if np.any(valid) else np.array([0.0])
    else:
        dq_dv = np.array([0.0])
    out: dict[str, Any] = {
        "battery": str(group["battery"].iloc[0]),
        "event_type": event_type,
        "event_type_discharge": 1.0 if event_type == "discharge" else 0.0,
        "event_order_type": int(group["event_order"].iloc[0]),
        "mat_step_index": int(group["mat_step_index"].iloc[0]),
        "date": str(group["date"].iloc[0]),
        "start_abs_time": float(group["time"].iloc[0]),
        "end_abs_time": float(group["time"].iloc[-1]),
        "label_charge_count": label_count,
        "assigned_checkpoint_cycle": int(checkpoint),
        "capacity_ah": float(capacity),
        "soh_percent": float(soh),
        "initial_capacity_ah": float(initial_capacity),
        "sample_count": int(len(group)),
        "duration_s": duration,
        "current_mean_a": float(np.mean(current)),
        "current_abs_mean_a": float(np.mean(np.abs(current))),
        "current_throughput_signed_ah": float(current_throughput_signed),
        "throughput_magnitude_ah": float(throughput_magnitude),
        "energy_signed_wh": float(energy_signed),
        "energy_magnitude_wh": float(energy_mag),
        "power_mean_signed_w": float(np.mean(power_signed)),
        "power_mean_magnitude_w": float(np.mean(power_mag)),
        "voltage_start": float(voltage[0]),
        "voltage_end": float(voltage[-1]),
        "voltage_delta_signed": float(voltage[-1] - voltage[0]),
        "dq_dv_max_ahpv": float(np.max(dq_dv)),
    }
    out.update(sequence_stats(voltage, "voltage_pdf"))
    return out


def build_or_load_event_table() -> pd.DataFrame:
    event_path = OUT / "charge_discharge_event_features_top20_source.csv"
    if event_path.exists():
        return pd.read_csv(event_path)
    rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    for battery in BATTERIES:
        checkpoint_counts, capacities, initial_capacity = load_reference_label_table(battery)
        for event_type in ("charge", "discharge"):
            path = SEGMENT_DIR / f"{battery}_{event_type}_random_walk_segments.pkl"
            df = pd.read_pickle(path)
            verification_rows.append(
                {
                    "battery": battery,
                    "event_type": event_type,
                    "rows": int(len(df)),
                    "events": int(df["event_order"].nunique()),
                    "mean_current_a": float(df["current"].mean()),
                    "min_current_a": float(df["current"].min()),
                    "max_current_a": float(df["current"].max()),
                }
            )
            event_order = df["event_order"].to_numpy(dtype=np.int64)
            unique_orders, starts = np.unique(event_order, return_index=True)
            ends = np.r_[starts[1:], len(df)]
            for idx, (start, end) in enumerate(zip(starts, ends), start=1):
                rows.append(build_one_event_features(df.iloc[start:end], checkpoint_counts, capacities, initial_capacity))
                if idx % 5000 == 0:
                    print(f"{battery} {event_type}: processed {idx}/{len(unique_orders)} events", flush=True)
    features = pd.DataFrame(rows).sort_values(["battery", "start_abs_time", "mat_step_index"]).reset_index(drop=True)
    features["event_order_chronological"] = features.groupby("battery").cumcount() + 1
    features.to_csv(event_path, index=False)
    pd.DataFrame(verification_rows).to_csv(OUT / "charge_discharge_verification.csv", index=False)
    return features


def add_block_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sizes = df.groupby(["battery", "assigned_checkpoint_cycle"])["soh_percent"].transform("count").astype(float)
    df["sample_weight"] = 1.0 / sizes
    df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()
    return df


def make_windows(bdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    bdf = bdf.sort_values("event_order_chronological").reset_index(drop=True)
    x = bdf[FEATURES].to_numpy(dtype=np.float32)
    y = bdf["soh_percent"].to_numpy(dtype=np.float32)
    weights = bdf["sample_weight"].to_numpy(dtype=np.float32)
    windows: list[np.ndarray] = []
    targets: list[float] = []
    w_out: list[float] = []
    rows: list[pd.Series] = []
    for end in range(CFG.window - 1, len(bdf)):
        start = end - CFG.window + 1
        windows.append(x[start : end + 1])
        targets.append(float(y[end]))
        w_out.append(float(weights[end]))
        rows.append(bdf.iloc[end])
    meta = pd.DataFrame(rows).reset_index(drop=True)
    return np.stack(windows), np.asarray(targets, dtype=np.float32), np.asarray(w_out, dtype=np.float32), meta["event_order_chronological"].to_numpy(dtype=np.int64), meta


def stratified_val_mask(y: np.ndarray, frac: float, seed: int, bins: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = np.linspace(float(y.min()) - 1e-6, float(y.max()) + 1e-6, bins + 1)
    bin_id = np.digitize(y, edges) - 1
    mask = np.zeros(len(y), dtype=bool)
    for b in range(bins):
        idx = np.flatnonzero(bin_id == b)
        if idx.size:
            n = max(1, int(round(idx.size * frac)))
            mask[rng.choice(idx, size=n, replace=False)] = True
    return mask


def fit_scaler(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    lo = np.quantile(flat, CFG.clip_quantile, axis=0).astype(np.float32)
    hi = np.quantile(flat, 1.0 - CFG.clip_quantile, axis=0).astype(np.float32)
    clipped = np.clip(flat, lo, hi)
    mean = clipped.mean(axis=0).astype(np.float32)
    std = clipped.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return lo, hi, mean, std


def apply_scaler(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.clip(x, lo, hi) - mean) / std).astype(np.float32)


def standardize_target(train_y: np.ndarray, *others: np.ndarray) -> tuple[list[np.ndarray], float, float]:
    mean = float(train_y.mean())
    std = float(train_y.std())
    if std < 1e-8:
        std = 1.0
    all_y = [train_y, *others]
    return [((y - mean) / std).astype(np.float32) for y in all_y], mean, std


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.w = w.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx]), torch.tensor(self.w[idx])


class EventTransformer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, CFG.d_model))
        self.input = nn.Sequential(nn.Linear(n_features, CFG.d_model), nn.LayerNorm(CFG.d_model))
        self.pos = nn.Parameter(torch.zeros(1, CFG.window + 1, CFG.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=CFG.d_model,
            nhead=CFG.n_heads,
            dim_feedforward=CFG.ffn,
            dropout=CFG.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=CFG.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(CFG.d_model),
            nn.Linear(CFG.d_model, CFG.d_model),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.input(x)
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.shape[1] + 1]
        encoded = self.encoder(seq)
        return self.head(encoded[:, 0]).squeeze(-1)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    loss = torch.where(abs_err <= delta, 0.5 * err.pow(2), delta * (abs_err - 0.5 * delta))
    return (loss * weight).sum() / torch.clamp(weight.sum(), min=1e-8)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    y_mean: float,
    y_std: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    losses: list[float] = []
    for xb, yb, wb in loader:
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = weighted_huber(out, yb, wb)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        losses.append(float(loss.detach().cpu()))
        preds.append((out.detach().cpu().numpy() * y_std + y_mean).astype(np.float32))
        trues.append((yb.detach().cpu().numpy() * y_std + y_mean).astype(np.float32))
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    metrics = regression_metrics(true, pred)
    metrics["loss"] = float(np.mean(losses))
    return metrics, pred, true


def aggregate_by_checkpoint(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    df = meta[["battery", "assigned_checkpoint_cycle"]].copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    agg = df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False).agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    out = regression_metrics(agg["y_true"].to_numpy(), agg["y_pred"].to_numpy())
    out["n_checkpoints"] = int(len(agg))
    agg.to_csv(OUT / "test_checkpoint_predictions.csv", index=False)
    return out


def prepare_data() -> dict[str, Any]:
    features = add_block_weights(build_or_load_event_table())
    feature_defs = []
    for idx, feat in enumerate(FEATURES, start=1):
        row = FEATURE_DEFS[feat].copy()
        row["rank"] = idx
        row["code_name"] = feat
        feature_defs.append(row)
    pd.DataFrame(feature_defs).to_csv(OUT / "top20_charge_discharge_engineered_features.csv", index=False)
    pd.DataFrame({"input_feature": FEATURES}).to_csv(OUT / "input_features_used.csv", index=False)

    split_rows: list[dict[str, Any]] = []
    xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    ws: dict[str, np.ndarray] = {}
    metas: dict[str, pd.DataFrame] = {}
    orders: dict[str, np.ndarray] = {}
    for battery in BATTERIES:
        bdf = features[features["battery"].eq(battery)].reset_index(drop=True)
        x, y, w, order, meta = make_windows(bdf)
        xs[battery], ys[battery], ws[battery], orders[battery], metas[battery] = x, y, w, order, meta
        split_rows.append(
            {
                "battery": battery,
                "events_total": int(len(bdf)),
                "charge_events": int(bdf["event_type"].eq("charge").sum()),
                "discharge_events": int(bdf["event_type"].eq("discharge").sum()),
                "model_windows": int(len(x)),
                "unique_reference_checkpoints": int(bdf["assigned_checkpoint_cycle"].nunique()),
                "current_mean_charge": float(bdf.loc[bdf["event_type"].eq("charge"), "current_mean_a"].mean()),
                "current_mean_discharge": float(bdf.loc[bdf["event_type"].eq("discharge"), "current_mean_a"].mean()),
            }
        )
    pd.DataFrame(split_rows).to_csv(OUT / "split_and_event_verification.csv", index=False)

    x9, y9, w9 = xs["RW9"], ys["RW9"], ws["RW9"]
    x10, y10, w10 = xs["RW10"], ys["RW10"], ws["RW10"]
    v9 = stratified_val_mask(y9, CFG.val_frac, CFG.seed)
    v10 = stratified_val_mask(y10, CFG.val_frac, CFG.seed + 1)
    x_train_raw = np.concatenate([x9[~v9], x10[~v10]], axis=0)
    y_train = np.concatenate([y9[~v9], y10[~v10]], axis=0)
    w_train = np.concatenate([w9[~v9], w10[~v10]], axis=0)
    x_val_raw = np.concatenate([x9[v9], x10[v10]], axis=0)
    y_val = np.concatenate([y9[v9], y10[v10]], axis=0)
    w_val = np.ones_like(y_val, dtype=np.float32)
    meta_val = pd.concat([metas["RW9"].loc[v9], metas["RW10"].loc[v10]], ignore_index=True)
    x_test_raw, y_test, w_test, meta_test = xs["RW11"], ys["RW11"], np.ones_like(ys["RW11"], dtype=np.float32), metas["RW11"]

    lo, hi, mean, std = fit_scaler(x_train_raw)
    x_train = apply_scaler(x_train_raw, lo, hi, mean, std)
    x_val = apply_scaler(x_val_raw, lo, hi, mean, std)
    x_test = apply_scaler(x_test_raw, lo, hi, mean, std)
    (y_train_std, y_val_std, y_test_std), y_mean, y_std = standardize_target(y_train, y_val, y_test)
    return {
        "features": features,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_std": y_train_std,
        "y_val_std": y_val_std,
        "y_test_std": y_test_std,
        "w_train": w_train,
        "w_val": w_val,
        "w_test": w_test,
        "meta_val": meta_val,
        "meta_test": meta_test,
        "scaler": {"lo": lo, "hi": hi, "mean": mean, "std": std, "y_mean": y_mean, "y_std": y_std},
    }


def train_transformer(data: dict[str, Any]) -> tuple[EventTransformer, dict[str, Any], dict[str, np.ndarray]]:
    train_loader = DataLoader(SeqDataset(data["x_train"], data["y_train_std"], data["w_train"]), batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(data["x_val"], data["y_val_std"], data["w_val"]), batch_size=CFG.batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(data["x_test"], data["y_test_std"], data["w_test"]), batch_size=CFG.batch_size, shuffle=False)
    model = EventTransformer(len(FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    total_steps = CFG.epochs * max(1, len(train_loader))
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_state = None
    best = {"val_r2": -1e9, "epoch": -1}
    history = []
    for epoch in range(1, CFG.epochs + 1):
        train_m, _, _ = run_epoch(model, train_loader, optimizer, scheduler, data["scaler"]["y_mean"], data["scaler"]["y_std"])
        with torch.no_grad():
            val_m, _, _ = run_epoch(model, val_loader, None, None, data["scaler"]["y_mean"], data["scaler"]["y_std"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_r2": train_m["r2"],
                "train_mae": train_m["mae"],
                "val_loss": val_m["loss"],
                "val_r2": val_m["r2"],
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
            }
        )
        pd.DataFrame(history).to_csv(OUT / "training_history_partial.csv", index=False)
        print(f"epoch {epoch:02d}/{CFG.epochs} train_r2={train_m['r2']:.4f} val_r2={val_m['r2']:.4f} val_mae={val_m['mae']:.3f}", flush=True)
        if val_m["r2"] > best["val_r2"]:
            best = {"val_r2": val_m["r2"], "val_mae": val_m["mae"], "val_rmse": val_m["rmse"], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        train_final, train_pred, train_true = run_epoch(model, train_loader, None, None, data["scaler"]["y_mean"], data["scaler"]["y_std"])
        val_final, val_pred, val_true = run_epoch(model, val_loader, None, None, data["scaler"]["y_mean"], data["scaler"]["y_std"])
        test_final, test_pred, test_true = run_epoch(model, test_loader, None, None, data["scaler"]["y_mean"], data["scaler"]["y_std"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT / "training_history.csv", index=False)
    pred_df = data["meta_test"].copy()
    pred_df["y_true_soh"] = test_true
    pred_df["y_pred_soh"] = test_pred
    pred_df["abs_error"] = np.abs(test_pred - test_true)
    pred_df.to_csv(OUT / "test_predictions.csv", index=False)
    val_pred_df = data["meta_val"].copy()
    val_pred_df["y_true_soh"] = val_true
    val_pred_df["y_pred_soh"] = val_pred
    val_pred_df.to_csv(OUT / "validation_predictions.csv", index=False)
    checkpoint_metrics = aggregate_by_checkpoint(data["meta_test"], test_true, test_pred)
    metrics = {
        "split": {
            "train_batteries": TRAIN_BATTERIES,
            "validation": "10% stratified windows from RW9 and RW10",
            "test_batteries": TEST_BATTERIES,
            "rw12_used": False,
        },
        "config": asdict(CFG),
        "features": FEATURES,
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_val": best,
        "train_event": train_final,
        "validation_event": val_final,
        "test_event": test_final,
        "test_checkpoint": checkpoint_metrics,
        "data_sizes": {
            "train_windows": int(len(data["x_train"])),
            "validation_windows": int(len(data["x_val"])),
            "test_windows": int(len(data["x_test"])),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(CFG),
            "features": FEATURES,
            "scaler": data["scaler"],
        },
        OUT / "model.pt",
    )
    arrays = {
        "train_pred": train_pred,
        "train_true": train_true,
        "val_pred": val_pred,
        "val_true": val_true,
        "test_pred": test_pred,
        "test_true": test_true,
    }
    return model, metrics, arrays


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_event_verification_plot() -> None:
    df = pd.read_csv(OUT / "split_and_event_verification.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    x = np.arange(len(df))
    width = 0.38
    axes[0].bar(x - width / 2, df["charge_events"], width, label="Charge events", color="#148778")
    axes[0].bar(x + width / 2, df["discharge_events"], width, label="Discharge events", color="#ea580c")
    axes[0].set_xticks(x, df["battery"])
    axes[0].set_ylabel("Number of events")
    axes[0].set_title("Verified Random-Walk Event Counts")
    axes[0].legend(frameon=False)
    style_axis(axes[0])
    axes[1].bar(x - width / 2, df["current_mean_charge"], width, label="Charge current mean", color="#148778")
    axes[1].bar(x + width / 2, df["current_mean_discharge"], width, label="Discharge current mean", color="#ea580c")
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_xticks(x, df["battery"])
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Current Sign Verification")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    fig.tight_layout()
    fig.savefig(FIG / "00_event_type_verification.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_result_plots(metrics: dict[str, Any]) -> None:
    hist = pd.read_csv(OUT / "training_history.csv")
    pred = pd.read_csv(OUT / "test_predictions.csv")
    plot_hist = hist[hist["epoch"] >= 3].copy()
    if len(plot_hist) < 3:
        plot_hist = hist
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.plot(plot_hist["epoch"], plot_hist["train_loss"], linewidth=2.8, label="Training loss", color="#2563eb")
    ax.plot(plot_hist["epoch"], plot_hist["val_loss"], linewidth=2.8, label="Validation loss", color="#ea580c")
    ax.set_title("Charge+Discharge Transformer: Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted Huber loss")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "01_loss.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.plot(plot_hist["epoch"], plot_hist["train_r2"].clip(lower=0.0), linewidth=2.8, label="Training R2", color="#2563eb")
    ax.plot(plot_hist["epoch"], plot_hist["val_r2"].clip(lower=0.0), linewidth=2.8, label="Validation R2", color="#ea580c")
    ax.axhline(0.90, color="#475569", linestyle=":", linewidth=2.2, label="R2 = 0.90")
    ax.axvline(metrics["best_val"]["epoch"], color="#148778", linestyle="--", linewidth=2.2, label=f"Best epoch = {metrics['best_val']['epoch']}")
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Charge+Discharge Transformer: Training and Validation R2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2")
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "02_r2.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    colors = pred["event_type"].map({"charge": "#148778", "discharge": "#ea580c"}).fillna("#64748b")
    ax.scatter(pred["y_true_soh"], pred["y_pred_soh"], s=12, alpha=0.28, c=colors, edgecolors="none")
    lo = float(min(pred["y_true_soh"].min(), pred["y_pred_soh"].min()))
    hi = float(max(pred["y_true_soh"].max(), pred["y_pred_soh"].max()))
    ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=2.2, label="Ideal")
    ax.set_title("RW11 Test: Predicted vs Actual SOH")
    ax.set_xlabel("Actual SOH (%)")
    ax.set_ylabel("Predicted SOH (%)")
    ax.text(
        0.04,
        0.96,
        f"R2 = {metrics['test_event']['r2']:.4f}\nMAE = {metrics['test_event']['mae']:.2f}%\nRMSE = {metrics['test_event']['rmse']:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"},
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "03_pred_vs_actual.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.plot(pred["event_order_chronological"], pred["y_true_soh"], color="#132b4a", linewidth=2.0, label="Actual SOH")
    ax.plot(pred["event_order_chronological"], pred["y_pred_soh"], color="#ea580c", linewidth=1.35, alpha=0.9, label="Predicted SOH")
    ax.set_title("RW11 Test: SOH Over Chronological Random-Walk Events")
    ax.set_xlabel("Chronological charge+discharge event index in RW11")
    ax.set_ylabel("SOH (%)")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "04_soh_trajectory.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def model_predict(model: nn.Module, x: np.ndarray, y_mean: float, y_std: float, batch: int = 2048) -> np.ndarray:
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.tensor(x[start : start + batch], dtype=torch.float32)
            outs.append((model(xb).detach().cpu().numpy() * y_std + y_mean).astype(np.float32))
    return np.concatenate(outs)


def make_xai(model: EventTransformer, data: dict[str, Any], metrics: dict[str, Any]) -> None:
    y_mean = data["scaler"]["y_mean"]
    y_std = data["scaler"]["y_std"]
    base_pred = model_predict(model, data["x_test"], y_mean, y_std)
    base_r2 = r2_score(data["y_test"], base_pred)
    rng = np.random.default_rng(CFG.seed)
    rows = []
    for j, feat in enumerate(FEATURES):
        x_perm = data["x_test"].copy()
        order = rng.permutation(x_perm.shape[0])
        x_perm[:, :, j] = x_perm[order, :, j]
        pred = model_predict(model, x_perm, y_mean, y_std)
        rows.append({"feature": feat, "label": FEATURE_DEFS[feat]["label"], "r2_drop": float(base_r2 - r2_score(data["y_test"], pred))})
    perm = pd.DataFrame(rows).sort_values("r2_drop", ascending=False).reset_index(drop=True)
    perm.to_csv(OUT / "global_permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    plot = perm.head(15).iloc[::-1]
    ax.barh(plot["label"], plot["r2_drop"], color="#148778")
    ax.set_title("Global XAI: Permutation Importance")
    ax.set_xlabel("RW11 test R2 drop when feature is shuffled")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG / "05_global_permutation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    sample_n = min(512, len(data["x_test"]))
    sample_idx = np.linspace(0, len(data["x_test"]) - 1, sample_n, dtype=int)
    x = torch.tensor(data["x_test"][sample_idx], dtype=torch.float32, requires_grad=True)
    model.zero_grad(set_to_none=True)
    out = model(x).sum()
    out.backward()
    attr = (x.grad.detach().numpy() * x.detach().numpy()) * y_std
    shap_like = np.mean(np.abs(attr), axis=(0, 1))
    shap_rows = pd.DataFrame({"feature": FEATURES, "label": [FEATURE_DEFS[f]["label"] for f in FEATURES], "mean_abs_gradient_shap": shap_like})
    shap_rows = shap_rows.sort_values("mean_abs_gradient_shap", ascending=False).reset_index(drop=True)
    shap_rows.to_csv(OUT / "global_shap_gradient_attribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    plot = shap_rows.head(15).iloc[::-1]
    ax.barh(plot["label"], plot["mean_abs_gradient_shap"], color="#2563eb")
    ax.set_title("Global SHAP-Style Attribution")
    ax.set_xlabel("Mean absolute gradient attribution across test windows")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG / "06_global_shap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    local_indices = choose_local_indices(data["meta_test"], data["y_test"], base_pred)
    local_rows = []
    for event_type, idx in local_indices.items():
        local_rows.extend(make_lime_plot(model, data, int(idx), event_type))
    pd.DataFrame(local_rows).to_csv(OUT / "local_lime_charge_discharge.csv", index=False)


def choose_local_indices(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    err = np.abs(y_true - y_pred)
    for event_type in ("charge", "discharge"):
        idx = np.flatnonzero(meta["event_type"].eq(event_type).to_numpy())
        ordered = idx[np.argsort(err[idx])]
        out[event_type] = int(ordered[len(ordered) // 2])
    return out


def make_lime_plot(model: EventTransformer, data: dict[str, Any], local_idx: int, event_type: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(CFG.seed + local_idx)
    base = data["x_test"][local_idx].copy()
    n = 700
    z = rng.normal(0.0, 0.55, size=(n, len(FEATURES))).astype(np.float32)
    x_pert = np.repeat(base[None, :, :], n, axis=0)
    for j in range(len(FEATURES)):
        x_pert[:, :, j] += z[:, j][:, None]
    pred_pert = model_predict(model, x_pert, data["scaler"]["y_mean"], data["scaler"]["y_std"], batch=1024)
    dist = np.sqrt((z**2).sum(axis=1))
    weights = np.exp(-(dist**2) / (2 * 2.0**2))
    ridge = Ridge(alpha=0.3)
    ridge.fit(z, pred_pert, sample_weight=weights)
    coefs = ridge.coef_
    pred = float(model_predict(model, data["x_test"][local_idx : local_idx + 1], data["scaler"]["y_mean"], data["scaler"]["y_std"])[0])
    actual = float(data["y_test"][local_idx])
    values = data["x_test_raw"][local_idx, -1, :]
    top = np.argsort(np.abs(coefs))[-10:][::-1]
    rows = []
    for rank, j in enumerate(top, start=1):
        rows.append(
            {
                "event_type": event_type,
                "rank": rank,
                "feature": FEATURES[j],
                "label": FEATURE_DEFS[FEATURES[j]]["label"],
                "coefficient": float(coefs[j]),
                "last_token_raw_value": float(values[j]),
                "predicted_soh": pred,
                "actual_soh": actual,
            }
        )

    fig = plt.figure(figsize=(13.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.8, 1.20], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    y_min, y_max = float(np.min(data["y_test"])), float(np.max(data["y_test"]))
    ax0.set_xlim(y_min, y_max)
    ax0.set_ylim(0, 1)
    ax0.barh([0.54], [pred - y_min], left=y_min, height=0.20, color="#f97316")
    ax0.axvline(actual, color="#2563eb", linewidth=2.3, label="Actual")
    ax0.axvline(pred, color="#111827", linewidth=2.3, label="Predicted")
    ax0.set_title(f"Local Prediction\n{event_type.capitalize()}-ending window", fontsize=15)
    ax0.set_xlabel("SOH (%)")
    ax0.set_yticks([])
    ax0.text(0.03, 0.18, f"Predicted = {pred:.2f}%\nActual = {actual:.2f}%", transform=ax0.transAxes, fontsize=12)
    ax0.legend(frameon=False, fontsize=9, loc="lower right")
    ax0.spines[["top", "right", "left"]].set_visible(False)

    selected = top[::-1]
    labels = [FEATURE_DEFS[FEATURES[i]]["label"] for i in selected]
    colors = ["#dc2626" if coefs[i] < 0 else "#f97316" for i in selected]
    ax1.barh(labels, coefs[selected], color=colors)
    ax1.axvline(0, color="#111827", linewidth=1)
    ax1.set_title("Local LIME Contributions", fontsize=15)
    ax1.set_xlabel("Surrogate coefficient")
    style_axis(ax1)

    ax2.axis("off")
    table_rows = [["Feature", "Value"]]
    for i in top[:6]:
        table_rows.append([FEATURE_DEFS[FEATURES[i]]["label"], f"{values[i]:.4g}"])
    table = ax2.table(cellText=table_rows, cellLoc="left", loc="center", colWidths=[0.75, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#132b4a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#e8f4fc" if r % 2 else "#f8fafc")
    fig.suptitle(f"Tabular LIME Explanation: RW11 {event_type.capitalize()} Example", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / f"07_local_lime_{event_type}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return rows


def add_textbox(slide, text: str, left: float, top: float, width: float, height: float, font_size: int = 16, bold: bool = False, color: RGBColor = BLACK):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.name = "Times New Roman"
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def title(slide, text: str, subtitle: str | None = None) -> None:
    add_textbox(slide, text, 0.45, 0.25, 12.4, 0.45, 26, True, NAVY)
    if subtitle:
        add_textbox(slide, subtitle, 0.48, 0.72, 12.2, 0.35, 12, False, SLATE)


def add_picture(slide, path: Path, left: float, top: float, width: float) -> None:
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


def add_table(slide, df: pd.DataFrame, left: float, top: float, width: float, height: float, font_size: int = 10) -> None:
    rows, cols = df.shape[0] + 1, df.shape[1]
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = shape.table
    for c, col in enumerate(df.columns):
        cell = table.cell(0, c)
        cell.text = str(col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text_frame.paragraphs[0].runs[0].font.color.rgb = WHITE
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
        cell.text_frame.paragraphs[0].runs[0].font.name = "Times New Roman"
        cell.text_frame.paragraphs[0].runs[0].font.size = Pt(font_size)
    for r in range(df.shape[0]):
        for c in range(cols):
            cell = table.cell(r + 1, c)
            cell.text = str(df.iat[r, c])
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BLUE if r % 2 == 0 else LIGHT_GRAY
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = BLACK


def build_ppt(metrics: dict[str, Any]) -> Path:
    ppt = Presentation()
    ppt.slide_width = Inches(13.333)
    ppt.slide_height = Inches(7.5)
    blank = ppt.slide_layouts[6]

    slide = ppt.slides.add_slide(blank)
    title(slide, "Charge+Discharge Transformer for Battery SOH", "RW9+RW10 training, stratified validation, RW11 held-out testing; RW12 unused.")
    add_textbox(slide, "Goal: predict SOH (%) from random-walk charge and discharge events without using capacity/SOH as an input.", 0.65, 1.35, 11.8, 0.45, 18, False)
    agenda = [
        "1. Verify charge and discharge events in the raw extracted data",
        "2. Define the 20 input features and what changed from charge-only",
        "3. Show transformer structure and training setup",
        "4. Show testing results and checkpoint aggregation",
        "5. Explain global and local behavior using permutation XAI, SHAP-style attribution, and LIME",
    ]
    for i, item in enumerate(agenda):
        add_textbox(slide, item, 0.85, 2.0 + i * 0.55, 11.8, 0.35, 16, False)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Data Verification", "Both random-walk charge and random-walk discharge events are present and used as inputs.")
    add_picture(slide, FIG / "00_event_type_verification.png", 0.6, 1.2, 12.0)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Input Feature Set: What Changed", "The charge-only features were made direction-safe for charge+discharge modeling.")
    changed = pd.read_csv(OUT / "top20_charge_discharge_engineered_features.csv")
    changed = changed[["label", "change"]].rename(columns={"label": "Feature label", "change": "Status / change"}).head(20)
    add_table(slide, changed.head(10), 0.4, 1.18, 6.1, 5.6, 9)
    add_table(slide, changed.tail(10), 6.8, 1.18, 6.1, 5.6, 9)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Feature Equations and Meanings", "Only Voltage, Current, relTime, and event direction are used; no capacity/SOH input.")
    feat_table = pd.read_csv(OUT / "top20_charge_discharge_engineered_features.csv")
    feat_table = feat_table[["label", "equation", "unit", "meaning"]].rename(columns={"label": "Feature", "equation": "Equation", "unit": "Unit", "meaning": "Physical meaning"})
    add_table(slide, feat_table.head(7), 0.25, 1.15, 12.85, 5.85, 8)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Feature Equations and Meanings Continued")
    add_table(slide, feat_table.iloc[7:14], 0.25, 1.0, 12.85, 6.1, 8)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Feature Equations and Meanings Continued")
    add_table(slide, feat_table.iloc[14:20], 0.25, 1.0, 12.85, 5.4, 8)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Transformer Structure and Training Setup")
    settings = pd.DataFrame(
        [
            ["Input sample", "32 chronological random-walk events"],
            ["Token", "One charge or discharge event summarized by 20 features"],
            ["Embedding", "Linear 20 -> 64 + learned CLS token"],
            ["Transformer", "2 encoder layers, 4 attention heads, FFN 128, GELU"],
            ["Regularization", "Dropout 0.10, weight decay 1e-4, gradient clipping 1.0"],
            ["Optimizer", "AdamW, learning rate 5e-4 with warmup + cosine decay"],
            ["Loss", "Inverse-checkpoint-block weighted Huber loss"],
            ["Train / validation / test", "RW9+RW10 / 10% stratified validation / full RW11"],
            ["Trainable parameters", f"{metrics['parameter_count']:,}"],
        ],
        columns=["Item", "Value"],
    )
    add_table(slide, settings, 0.7, 1.2, 11.9, 5.7, 11)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Training Behavior")
    add_picture(slide, FIG / "01_loss.png", 0.55, 1.12, 6.05)
    add_picture(slide, FIG / "02_r2.png", 6.75, 1.12, 6.05)

    slide = ppt.slides.add_slide(blank)
    title(slide, "RW11 Testing Results")
    add_picture(slide, FIG / "03_pred_vs_actual.png", 0.45, 1.08, 5.6)
    results = pd.DataFrame(
        [
            ["Event-level R2", f"{metrics['test_event']['r2']:.4f}"],
            ["Event-level MAE", f"{metrics['test_event']['mae']:.3f}% SOH"],
            ["Event-level RMSE", f"{metrics['test_event']['rmse']:.3f}% SOH"],
            ["Checkpoint R2", f"{metrics['test_checkpoint']['r2']:.4f}"],
            ["Checkpoint MAE", f"{metrics['test_checkpoint']['mae']:.3f}% SOH"],
            ["Checkpoint RMSE", f"{metrics['test_checkpoint']['rmse']:.3f}% SOH"],
            ["Test windows", f"{metrics['data_sizes']['test_windows']:,}"],
        ],
        columns=["Metric", "Value"],
    )
    add_table(slide, results, 6.55, 1.35, 6.2, 4.8, 13)

    slide = ppt.slides.add_slide(blank)
    title(slide, "RW11 SOH Trajectory")
    add_picture(slide, FIG / "04_soh_trajectory.png", 0.55, 1.1, 12.2)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Global XAI")
    add_picture(slide, FIG / "05_global_permutation.png", 0.4, 1.08, 6.25)
    add_picture(slide, FIG / "06_global_shap.png", 6.85, 1.08, 6.05)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Local LIME: Charge Example")
    add_picture(slide, FIG / "07_local_lime_charge.png", 0.35, 1.05, 12.5)

    slide = ppt.slides.add_slide(blank)
    title(slide, "Local LIME: Discharge Example")
    add_picture(slide, FIG / "07_local_lime_discharge.png", 0.35, 1.05, 12.5)

    out = OUT / "charge_discharge_top20_transformer_xai_report.pptx"
    ppt.save(out)
    return out


def main() -> None:
    set_seed(CFG.seed)
    data = prepare_data()
    make_event_verification_plot()
    model, metrics, _ = train_transformer(data)
    make_result_plots(metrics)
    make_xai(model, data, metrics)
    ppt_path = build_ppt(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"PPT saved to: {ppt_path}")


if __name__ == "__main__":
    main()
