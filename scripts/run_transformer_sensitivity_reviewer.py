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
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

import run_label_strategy_comparison as labels


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reviewer_transformer_sensitivity"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
BATTERIES = ["RW9", "RW10", "RW11"]

ALL_FEATURES = labels.FEATURES
VOLTAGE_FEATURES = [name for name in ALL_FEATURES if name.startswith("voltage_")]
LOADING_FEATURES = [
    "event_type_discharge",
    "current_mean_a",
    "current_abs_mean_a",
    "current_throughput_signed_ah",
    "throughput_magnitude_ah",
    "energy_signed_wh",
    "energy_magnitude_wh",
    "power_mean_signed_w",
    "power_mean_magnitude_w",
]


@dataclass(frozen=True)
class TrainingConfig:
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
    seed: int = 42


CFG = TrainingConfig()

EXPERIMENTS = [
    {"name": "window_1", "window": 1, "features": ALL_FEATURES, "target": "target_next_reference_soh", "weighted": True, "family": "window"},
    {"name": "window_8", "window": 8, "features": ALL_FEATURES, "target": "target_next_reference_soh", "weighted": True, "family": "window"},
    {"name": "window_16", "window": 16, "features": ALL_FEATURES, "target": "target_next_reference_soh", "weighted": True, "family": "window"},
    {"name": "window_64", "window": 64, "features": ALL_FEATURES, "target": "target_next_reference_soh", "weighted": True, "family": "window"},
    {"name": "dq_dv_removed", "window": 32, "features": [name for name in ALL_FEATURES if name != "dq_dv_max_ahpv"], "target": "target_next_reference_soh", "weighted": True, "family": "feature"},
    {"name": "voltage_only", "window": 32, "features": VOLTAGE_FEATURES, "target": "target_next_reference_soh", "weighted": True, "family": "feature"},
    {"name": "loading_only", "window": 32, "features": LOADING_FEATURES, "target": "target_next_reference_soh", "weighted": True, "family": "feature"},
    {"name": "no_block_weighting", "window": 32, "features": ALL_FEATURES, "target": "target_next_reference_soh", "weighted": False, "family": "feature"},
    {"name": "previous_reference", "window": 32, "features": ALL_FEATURES, "target": "target_previous_reference_soh", "weighted": True, "family": "label"},
    {"name": "linear_charge_count", "window": 32, "features": ALL_FEATURES, "target": "target_linear_charge_count_soh", "weighted": True, "family": "label"},
    {"name": "throughput_interpolation", "window": 32, "features": ALL_FEATURES, "target": "target_throughput_interpolation_soh", "weighted": True, "family": "label"},
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def stratified_mask(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = np.linspace(float(y.min()) - 1.0e-6, float(y.max()) + 1.0e-6, 11)
    groups = np.digitize(y, edges) - 1
    mask = np.zeros(len(y), dtype=bool)
    for group in range(10):
        indices = np.flatnonzero(groups == group)
        if len(indices):
            count = max(1, int(round(len(indices) * CFG.val_frac)))
            mask[rng.choice(indices, size=count, replace=False)] = True
    return mask


class LazyDataset(Dataset):
    def __init__(self, arrays: list[np.ndarray], battery_ids: np.ndarray, ends: np.ndarray, y: np.ndarray, w: np.ndarray, window: int):
        self.arrays = arrays
        self.battery_ids = battery_ids.astype(np.int8, copy=False)
        self.ends = ends.astype(np.int32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.w = w.astype(np.float32, copy=False)
        self.window = window

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        battery = int(self.battery_ids[index])
        end = int(self.ends[index])
        x = self.arrays[battery][end - self.window + 1 : end + 1]
        return torch.from_numpy(x), torch.tensor(self.y[index]), torch.tensor(self.w[index])


class EventTransformer(nn.Module):
    def __init__(self, n_features: int, window: int):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, CFG.d_model))
        self.input = nn.Sequential(nn.Linear(n_features, CFG.d_model), nn.LayerNorm(CFG.d_model))
        self.pos = nn.Parameter(torch.zeros(1, window + 1, CFG.d_model))
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
        self.head = nn.Sequential(nn.LayerNorm(CFG.d_model), nn.Linear(CFG.d_model, CFG.d_model), nn.GELU(), nn.Dropout(CFG.dropout), nn.Linear(CFG.d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.shape[1] + 1])
        return self.head(encoded[:, 0]).squeeze(-1)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    error = pred - target
    absolute = error.abs()
    loss = torch.where(absolute <= 1.0, 0.5 * error.pow(2), absolute - 0.5)
    return (loss * weight).sum() / torch.clamp(weight.sum(), min=1.0e-8)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, scheduler: Any, y_mean: float, y_std: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    predictions = []
    targets = []
    for x, y, w in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        batch_predictions = []
        total_weight = torch.clamp(w.sum(), min=1.0e-8)
        # Preserve the 4096-window effective batch while avoiding the quadratic
        # attention-memory spike of evaluating every 64-event sequence at once.
        for start in range(0, len(x), 256):
            stop = min(start + 256, len(x))
            prediction = model(x[start:stop])
            if training:
                chunk_loss = weighted_huber(prediction, y[start:stop], w[start:stop])
                (chunk_loss * (w[start:stop].sum() / total_weight)).backward()
            batch_predictions.append(prediction.detach())
        if training:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        prediction = torch.cat(batch_predictions)
        predictions.append((prediction.numpy() * y_std + y_mean).astype(np.float32))
        targets.append((y.detach().numpy() * y_std + y_mean).astype(np.float32))
    pred = np.concatenate(predictions)
    true = np.concatenate(targets)
    return metrics(true, pred), pred, true


def load_events() -> pd.DataFrame:
    frame = labels.interpolate_labels(labels.load_events())
    count = frame.groupby(["battery", "assigned_checkpoint_cycle"])["soh_percent"].transform("count").astype(float)
    frame["sample_weight"] = (1.0 / count) / (1.0 / count).mean()
    return frame


def prepare(events: pd.DataFrame, experiment: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    features = experiment["features"]
    window = experiment["window"]
    target = experiment["target"]
    blocks = [events[events["battery"].eq(name)].sort_values("event_order_chronological").reset_index(drop=True) for name in BATTERIES]
    raw = [block[features].to_numpy(dtype=np.float32) for block in blocks]
    ends = [np.arange(window - 1, len(block), dtype=np.int32) for block in blocks]
    y = [block[target].to_numpy(dtype=np.float32)[end] for block, end in zip(blocks, ends)]
    measured = [block["target_next_reference_soh"].to_numpy(dtype=np.float32)[end] for block, end in zip(blocks, ends)]
    weights = [block["sample_weight"].to_numpy(dtype=np.float32)[end] for block, end in zip(blocks, ends)]
    val_masks = [stratified_mask(y[index], CFG.seed + index) for index in range(2)]

    lo = np.empty(len(features), dtype=np.float32)
    hi = np.empty(len(features), dtype=np.float32)
    mean = np.empty(len(features), dtype=np.float32)
    std = np.empty(len(features), dtype=np.float32)
    for feature_index in range(len(features)):
        values = []
        for battery_index in range(2):
            view = np.lib.stride_tricks.sliding_window_view(raw[battery_index][:, feature_index], window)
            values.append(view[~val_masks[battery_index]].reshape(-1))
        joined = np.concatenate(values).astype(np.float32, copy=False)
        lo[feature_index] = np.quantile(joined, CFG.clip_quantile)
        hi[feature_index] = np.quantile(joined, 1.0 - CFG.clip_quantile)
        clipped = np.clip(joined, lo[feature_index], hi[feature_index])
        mean[feature_index] = clipped.mean()
        std[feature_index] = max(float(clipped.std()), 1.0e-8)
    normalized = [((np.clip(array, lo, hi) - mean) / std).astype(np.float32) for array in raw]

    train_y = np.concatenate([y[index][~val_masks[index]] for index in range(2)]).astype(np.float32)
    y_mean = float(train_y.mean())
    y_std = max(float(train_y.std()), 1.0e-8)

    def dataset(indices: list[int], validation: bool | None) -> LazyDataset:
        battery_ids = []
        selected_ends = []
        selected_y = []
        selected_w = []
        for battery_index in indices:
            if validation is None:
                mask = np.ones(len(ends[battery_index]), dtype=bool)
            else:
                mask = val_masks[battery_index] if validation else ~val_masks[battery_index]
            battery_ids.append(np.full(mask.sum(), battery_index, dtype=np.int8))
            selected_ends.append(ends[battery_index][mask])
            selected_y.append(((y[battery_index][mask] - y_mean) / y_std).astype(np.float32))
            if validation is False and experiment["weighted"]:
                selected_w.append(weights[battery_index][mask])
            else:
                selected_w.append(np.ones(mask.sum(), dtype=np.float32))
        return LazyDataset(normalized, np.concatenate(battery_ids), np.concatenate(selected_ends), np.concatenate(selected_y), np.concatenate(selected_w), window)

    split = {
        "train": dataset([0, 1], False),
        "val": dataset([0, 1], True),
        "test": dataset([2], None),
        "y_mean": y_mean,
        "y_std": y_std,
        "measured_test": measured[2],
        "strategy_test": y[2],
    }
    return split, blocks[2].iloc[ends[2]].reset_index(drop=True)


def aggregate(meta: pd.DataFrame, measured: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    frame = meta[["battery", "assigned_checkpoint_cycle"]].copy()
    frame["measured"] = measured
    frame["prediction"] = prediction
    grouped = frame.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False).agg(measured=("measured", "mean"), prediction=("prediction", "mean"))
    result = metrics(grouped["measured"].to_numpy(), grouped["prediction"].to_numpy())
    result["n_checkpoints"] = len(grouped)
    return result


def train_experiment(events: pd.DataFrame, experiment: dict[str, Any]) -> dict[str, Any]:
    set_seed(CFG.seed)
    directory = OUT / experiment["name"]
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        print(f"Using completed {experiment['name']}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))["row"]

    split, meta = prepare(events, experiment)
    train_loader = DataLoader(split["train"], batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(split["val"], batch_size=CFG.batch_size, shuffle=False)
    test_loader = DataLoader(split["test"], batch_size=CFG.batch_size, shuffle=False)
    model = EventTransformer(len(experiment["features"]), experiment["window"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    total_steps = CFG.epochs * max(1, len(train_loader))
    warmup = max(1, int(0.05 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best = {"r2": -1.0e9, "epoch": -1}
    best_state = None
    history = []
    resume_path = directory / "resume_checkpoint.pt"
    start_epoch = 1
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        best = resume["best"]
        best_state = resume["best_state"]
        history = resume["history"]
        start_epoch = int(resume["epoch"]) + 1
        print(f"Resuming {experiment['name']} at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, CFG.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, optimizer, scheduler, split["y_mean"], split["y_std"])
        with torch.no_grad():
            val_metrics, _, _ = run_epoch(model, val_loader, None, None, split["y_mean"], split["y_std"])
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in val_metrics.items()}})
        pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
        print(f"{experiment['name']} {epoch:02d}/{CFG.epochs} val_r2={val_metrics['r2']:.4f} val_rmse={val_metrics['rmse']:.3f}", flush=True)
        if val_metrics["r2"] > best["r2"]:
            best = {"r2": val_metrics["r2"], "epoch": epoch}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best": best,
                "best_state": best_state,
                "history": history,
            },
            resume_path,
        )
    if best_state is None:
        raise RuntimeError(f"No checkpoint for {experiment['name']}")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best": best}, directory / "best_model.pt")
    with torch.no_grad():
        strategy_metrics, prediction, strategy_true = run_epoch(model, test_loader, None, None, split["y_mean"], split["y_std"])
    measured_metrics = metrics(split["measured_test"], prediction)
    checkpoint_metrics = aggregate(meta, split["measured_test"], prediction)
    row = {
        "experiment": experiment["name"],
        "family": experiment["family"],
        "window": experiment["window"],
        "features": "+".join(experiment["features"]),
        "n_features": len(experiment["features"]),
        "target": experiment["target"],
        "weighted": experiment["weighted"],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "best_val_epoch": best["epoch"],
        **{f"strategy_event_{key}": value for key, value in strategy_metrics.items()},
        **{f"diagnostic_event_{key}": value for key, value in measured_metrics.items()},
        **{f"diagnostic_checkpoint_{key}": value for key, value in checkpoint_metrics.items()},
    }
    metrics_path.write_text(json.dumps({"config": asdict(CFG), "experiment": experiment, "row": row}, indent=2), encoding="utf-8")
    predictions = meta[["battery", "event_order_chronological", "assigned_checkpoint_cycle"]].copy()
    predictions["strategy_target"] = strategy_true
    predictions["diagnostic_target"] = split["measured_test"]
    predictions["prediction"] = prediction
    predictions.to_csv(directory / "test_predictions.csv", index=False)
    resume_path.unlink(missing_ok=True)
    return row


def add_reference_rows(frame: pd.DataFrame) -> pd.DataFrame:
    transformer = json.loads((ROOT / "reviewer_full_transformer_loco" / "test_RW11" / "metrics.json").read_text(encoding="utf-8"))["row"]
    reference = {
        "experiment": "complete_features_window_32",
        "family": "reference",
        "window": 32,
        "features": "+".join(ALL_FEATURES),
        "n_features": len(ALL_FEATURES),
        "target": "target_next_reference_soh",
        "weighted": True,
        "parameter_count": transformer["parameter_count"],
        "best_val_epoch": transformer["best_val_epoch"],
        "strategy_event_r2": transformer["event_r2"],
        "strategy_event_mae": transformer["event_mae"],
        "strategy_event_rmse": transformer["event_rmse"],
        "diagnostic_event_r2": transformer["event_r2"],
        "diagnostic_event_mae": transformer["event_mae"],
        "diagnostic_event_rmse": transformer["event_rmse"],
        "diagnostic_checkpoint_r2": transformer["checkpoint_r2"],
        "diagnostic_checkpoint_mae": transformer["checkpoint_mae"],
        "diagnostic_checkpoint_rmse": transformer["checkpoint_rmse"],
        "diagnostic_checkpoint_n_checkpoints": transformer["checkpoint_n_checkpoints"],
    }
    return pd.concat([pd.DataFrame([reference]), frame], ignore_index=True)


def plot_family(frame: pd.DataFrame, family: str, order: list[str], labels_out: list[str], filename: str) -> None:
    subset = frame.set_index("experiment").loc[order].reset_index()
    values = subset["diagnostic_checkpoint_rmse"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    bars = ax.bar(labels_out, values, color="#2f80ed", edgecolor="#263238")
    upper = values.max() * 1.18
    ax.set_ylim(0, upper)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + upper * 0.02, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylabel("Diagnostic-checkpoint RMSE (SOH percentage points)")
    ax.set_title(f"Transformer {family}", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    events = load_events()
    rows = []
    for experiment in EXPERIMENTS:
        rows.append(train_experiment(events, experiment))
        add_reference_rows(pd.DataFrame(rows)).to_csv(TABLES / "transformer_sensitivity_metrics_partial.csv", index=False)
    frame = add_reference_rows(pd.DataFrame(rows))
    frame.to_csv(TABLES / "transformer_sensitivity_metrics.csv", index=False)
    plot_family(frame, "Window-Length Sensitivity", ["window_1", "window_8", "window_16", "complete_features_window_32", "window_64"], ["1", "8", "16", "32", "64"], "fig_transformer_window_sensitivity.png")
    plot_family(frame, "Feature and Weighting Ablation", ["complete_features_window_32", "dq_dv_removed", "voltage_only", "loading_only", "no_block_weighting"], ["Complete", "No dQ/dV", "Voltage only", "Loading only", "No weighting"], "fig_transformer_feature_ablation.png")
    plot_family(frame, "Label-Strategy Sensitivity", ["complete_features_window_32", "previous_reference", "linear_charge_count", "throughput_interpolation"], ["Next ref.", "Previous ref.", "Charge interpolation", "Ah interpolation"], "fig_transformer_label_sensitivity.png")
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
