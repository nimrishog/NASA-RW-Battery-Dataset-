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


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "modeling" / "event_features_rw9_rw11.csv"
OUT = ROOT / "results" / "reviewer_validation" / "full_transformer_loco"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
OVERLEAF_FIGS = ROOT / "results" / "manuscript_figures"

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


@dataclass(frozen=True)
class Config:
    window: int = 32
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


CFG = Config()


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


def load_events() -> pd.DataFrame:
    cols = [
        "battery",
        "event_type",
        "event_order_chronological",
        "assigned_checkpoint_cycle",
        "label_charge_count",
        "soh_percent",
        *FEATURES,
    ]
    df = pd.read_csv(SOURCE, usecols=cols)
    df = df[df["battery"].isin(BATTERIES)].copy()
    df = df.sort_values(["battery", "event_order_chronological"]).reset_index(drop=True)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    for _, idx in df.groupby("battery").groups.items():
        block = df.loc[idx, FEATURES]
        df.loc[idx, FEATURES] = block.fillna(block.median(numeric_only=True)).fillna(0.0)
    sizes = df.groupby(["battery", "assigned_checkpoint_cycle"])["soh_percent"].transform("count").astype(float)
    df["sample_weight"] = 1.0 / sizes
    df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()
    return df


def make_windows(bdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
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
    return (
        np.stack(windows),
        np.asarray(targets, dtype=np.float32),
        np.asarray(w_out, dtype=np.float32),
        meta,
    )


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
    return [((y - mean) / std).astype(np.float32) for y in (train_y, *others)], mean, std


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


def aggregate_by_checkpoint(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> dict[str, float]:
    df = meta[["battery", "assigned_checkpoint_cycle"]].copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    agg = df.groupby(["battery", "assigned_checkpoint_cycle"], as_index=False).agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    agg.to_csv(out_path, index=False)
    out = regression_metrics(agg["y_true"].to_numpy(), agg["y_pred"].to_numpy())
    out["n_checkpoints"] = int(len(agg))
    return out


def prepare_split(events: pd.DataFrame, test_battery: str) -> dict[str, Any]:
    per_battery: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for battery in BATTERIES:
        per_battery[battery] = make_windows(events[events["battery"].eq(battery)].reset_index(drop=True))

    train_batteries = [b for b in BATTERIES if b != test_battery]
    train_parts = []
    val_parts = []
    for offset, battery in enumerate(train_batteries):
        x, y, w, meta = per_battery[battery]
        val_mask = stratified_val_mask(y, CFG.val_frac, CFG.seed + offset)
        train_parts.append((x[~val_mask], y[~val_mask], w[~val_mask]))
        val_parts.append((x[val_mask], y[val_mask], np.ones_like(y[val_mask], dtype=np.float32)))
    x_train_raw = np.concatenate([p[0] for p in train_parts], axis=0)
    y_train = np.concatenate([p[1] for p in train_parts], axis=0)
    w_train = np.concatenate([p[2] for p in train_parts], axis=0)
    x_val_raw = np.concatenate([p[0] for p in val_parts], axis=0)
    y_val = np.concatenate([p[1] for p in val_parts], axis=0)
    w_val = np.concatenate([p[2] for p in val_parts], axis=0)
    x_test_raw, y_test, _w_test, meta_test = per_battery[test_battery]
    w_test = np.ones_like(y_test, dtype=np.float32)

    lo, hi, mean, std = fit_scaler(x_train_raw)
    x_train = apply_scaler(x_train_raw, lo, hi, mean, std)
    x_val = apply_scaler(x_val_raw, lo, hi, mean, std)
    x_test = apply_scaler(x_test_raw, lo, hi, mean, std)
    (y_train_std, y_val_std, y_test_std), y_mean, y_std = standardize_target(y_train, y_val, y_test)
    return {
        "train_batteries": train_batteries,
        "test_battery": test_battery,
        "x_train": x_train,
        "y_train": y_train,
        "y_train_std": y_train_std,
        "w_train": w_train,
        "x_val": x_val,
        "y_val": y_val,
        "y_val_std": y_val_std,
        "w_val": w_val,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_std": y_test_std,
        "w_test": w_test,
        "meta_test": meta_test,
        "scaler": {"y_mean": y_mean, "y_std": y_std},
    }


def train_one_loco(events: pd.DataFrame, test_battery: str) -> dict[str, Any]:
    set_seed(CFG.seed)
    split = prepare_split(events, test_battery)
    out_dir = OUT / f"test_{test_battery}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(SeqDataset(split["x_train"], split["y_train_std"], split["w_train"]), batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(split["x_val"], split["y_val_std"], split["w_val"]), batch_size=CFG.batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(split["x_test"], split["y_test_std"], split["w_test"]), batch_size=CFG.batch_size, shuffle=False)
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
    checkpoint_path = out_dir / "best_model.pt"
    best_state = None
    best = {"val_r2": -1e9, "epoch": -1}
    history = []
    start_epoch = 1
    if checkpoint_path.exists() and (out_dir / "training_history.csv").exists():
        history = pd.read_csv(out_dir / "training_history.csv").to_dict("records")
        if len(history) >= CFG.epochs:
            saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_state = saved_checkpoint["state_dict"]
            best = saved_checkpoint["best"]
            start_epoch = CFG.epochs + 1
            print(f"Using completed training checkpoint for held-out {test_battery}", flush=True)
    for epoch in range(start_epoch, CFG.epochs + 1):
        train_m, _, _ = run_epoch(model, train_loader, optimizer, scheduler, split["scaler"]["y_mean"], split["scaler"]["y_std"])
        with torch.no_grad():
            val_m, _, _ = run_epoch(model, val_loader, None, None, split["scaler"]["y_mean"], split["scaler"]["y_std"])
        row = {
            "test_battery": test_battery,
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_r2": train_m["r2"],
            "train_mae": train_m["mae"],
            "val_loss": val_m["loss"],
            "val_r2": val_m["r2"],
            "val_mae": val_m["mae"],
            "val_rmse": val_m["rmse"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        print(f"{test_battery} epoch {epoch:02d}/{CFG.epochs} train_r2={train_m['r2']:.4f} val_r2={val_m['r2']:.4f} val_mae={val_m['mae']:.3f}", flush=True)
        if val_m["r2"] > best["val_r2"]:
            best = {"val_r2": val_m["r2"], "val_mae": val_m["mae"], "val_rmse": val_m["rmse"], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "best": best}, checkpoint_path)
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        train_final, _train_pred, _train_true = run_epoch(model, train_loader, None, None, split["scaler"]["y_mean"], split["scaler"]["y_std"])
        val_final, _val_pred, _val_true = run_epoch(model, val_loader, None, None, split["scaler"]["y_mean"], split["scaler"]["y_std"])
        test_final, test_pred, test_true = run_epoch(model, test_loader, None, None, split["scaler"]["y_mean"], split["scaler"]["y_std"])

    pred_df = split["meta_test"][["battery", "event_order_chronological", "assigned_checkpoint_cycle"]].copy()
    pred_df["y_true_soh"] = test_true
    pred_df["y_pred_soh"] = test_pred
    pred_df["abs_error"] = np.abs(test_pred - test_true)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    checkpoint = aggregate_by_checkpoint(split["meta_test"], test_true, test_pred, out_dir / "test_checkpoint_predictions.csv")
    row = {
        "experiment": "leave_one_cell_out_full_transformer",
        "model": "event_window_transformer",
        "train_batteries": "+".join(split["train_batteries"]),
        "test_battery": test_battery,
        "window": CFG.window,
        "test_windows": int(len(test_true)),
        "n_features": len(FEATURES),
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_val_epoch": int(best["epoch"]),
        **{f"event_{k}": v for k, v in test_final.items()},
        **{f"checkpoint_{k}": v for k, v in checkpoint.items()},
        "train_event_r2": train_final["r2"],
        "validation_event_r2": val_final["r2"],
    }
    (out_dir / "metrics.json").write_text(json.dumps({"config": asdict(CFG), "features": FEATURES, "row": row}, indent=2), encoding="utf-8")
    return row


def plot_loco(df: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = ["#2f80ed", "#1b998b", "#ef8a17"]
    ax.bar(df["test_battery"], df["checkpoint_rmse"], color=colors, edgecolor="#263238")
    for idx, value in enumerate(df["checkpoint_rmse"]):
        ax.text(idx, value + max(df["checkpoint_rmse"]) * 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Leave-One-Cell-Out Transformer RMSE", fontsize=17, weight="bold")
    ax.set_ylabel("Checkpoint RMSE (SOH percentage points)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_full_transformer_loco_rmse.png", dpi=300)
    OVERLEAF_FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OVERLEAF_FIGS / "fig_full_transformer_loco_rmse.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    events = load_events()
    rows = []
    for battery in BATTERIES:
        completed_metrics = OUT / f"test_{battery}" / "metrics.json"
        if completed_metrics.exists():
            saved = json.loads(completed_metrics.read_text(encoding="utf-8"))["row"]
            print(f"Using completed full Transformer LOCO fold for held-out {battery}", flush=True)
            rows.append(saved)
            pd.DataFrame(rows).to_csv(TABLES / "full_transformer_loco_metrics.csv", index=False)
            continue
        print(f"Running full Transformer LOCO with held-out {battery}", flush=True)
        rows.append(train_one_loco(events, battery))
        pd.DataFrame(rows).to_csv(TABLES / "full_transformer_loco_metrics.csv", index=False)
    df = pd.DataFrame(rows).sort_values("test_battery")
    df.to_csv(TABLES / "full_transformer_loco_metrics.csv", index=False)
    plot_loco(df)
    manifest = {
        "source": str(SOURCE),
        "config": asdict(CFG),
        "features": FEATURES,
        "outputs": [str(p) for p in sorted(OUT.rglob("*")) if p.is_file()],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
