from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

import run_charge_discharge_full_cycle_suite as suite


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
SUITE_DIR = ROOT / "charge_discharge_full_cycle_suite"
OUTDIR = SUITE_DIR / "interpretability_shap_lime"
TRANSFORMER_DIR = SUITE_DIR / "feature_tokens_h4_d64_l2"
TRAIN_BATTERIES = ("RW9", "RW10")
TEST_BATTERIES = ("RW11",)
RANDOM_SEED = 42
BACKGROUND_SIZE = 128
SHAP_SAMPLES_PER_EVENT_TYPE = 200
LIME_NUM_FEATURES = 15
LIME_NUM_SAMPLES = 5000


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data() -> dict[str, object]:
    features, _ = suite.build_or_load_event_table_and_patches()
    features = suite.add_block_weights(features.rename(columns={"label_charge_count": "charge_cycle_for_label"}))
    features = features.rename(columns={"charge_cycle_for_label": "label_charge_count"})

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
    train_mask = features["battery"].isin(TRAIN_BATTERIES).to_numpy()
    test_mask = features["battery"].isin(TEST_BATTERIES).to_numpy()
    train_meta = features.loc[train_mask].reset_index(drop=True)
    test_meta = features.loc[test_mask].reset_index(drop=True)
    train_raw = np.nan_to_num(train_meta[feature_names].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    test_raw = np.nan_to_num(test_meta[feature_names].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    train_x, test_x, feature_mean, feature_std = suite.standardize_matrix(train_raw, test_raw)
    train_y = train_meta["soh_percent"].to_numpy(dtype=np.float32)
    test_y = test_meta["soh_percent"].to_numpy(dtype=np.float32)
    train_w = train_meta["sample_weight"].to_numpy(dtype=np.float32)
    train_y_norm, _, y_mean, y_std = suite.standardize_target(train_y, test_y)
    return {
        "feature_names": feature_names,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y,
        "test_y": test_y,
        "train_w": train_w,
        "train_y_norm": train_y_norm,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


class SohWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, y_mean: float, y_std: float):
        super().__init__()
        self.base_model = base_model
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.base_model(x) * self.y_std + self.y_mean).unsqueeze(-1)


def load_transformer(feature_count: int, y_mean: float, y_std: float) -> SohWrapper:
    ckpt = torch.load(TRANSFORMER_DIR / "model.pt", map_location="cpu", weights_only=False)
    spec = suite.TransformerSpec(**ckpt["spec"])
    model = suite.FeatureTokenTransformer(feature_count, spec)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    wrapped = SohWrapper(model, y_mean, y_std)
    wrapped.eval()
    return wrapped


def train_lightgbm(train_raw: np.ndarray, train_y: np.ndarray, train_w: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.75,
        colsample_bytree=0.60,
        reg_lambda=5.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_raw, train_y, sample_weight=train_w)
    return model


def transformer_predict_fn(model: nn.Module) -> Callable[[np.ndarray], np.ndarray]:
    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                batch = torch.tensor(x[start : start + 4096], dtype=torch.float32)
                outs.append(model(batch).detach().cpu().numpy())
        return np.concatenate(outs).reshape(-1)

    return predict


def choose_indices(test_meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    abs_err = np.abs(y_pred - y_true)
    for event_type in ("charge", "discharge"):
        idx = np.flatnonzero(test_meta["event_type"].eq(event_type).to_numpy())
        ordered = idx[np.argsort(abs_err[idx])]
        out[event_type] = int(ordered[len(ordered) // 2])
    return out


def stratified_sample_indices(test_meta: pd.DataFrame, per_event_type: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    picks: list[np.ndarray] = []
    for event_type in ("charge", "discharge"):
        idx = np.flatnonzero(test_meta["event_type"].eq(event_type).to_numpy())
        size = min(per_event_type, len(idx))
        picks.append(rng.choice(idx, size=size, replace=False))
    return np.concatenate(picks)


def background_indices(train_meta: pd.DataFrame, size: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    picks: list[np.ndarray] = []
    per_type = max(1, size // 2)
    for event_type in ("charge", "discharge"):
        idx = np.flatnonzero(train_meta["event_type"].eq(event_type).to_numpy())
        picks.append(rng.choice(idx, size=min(per_type, len(idx)), replace=False))
    out = np.concatenate(picks)
    if len(out) < size:
        remaining = np.setdiff1d(np.arange(len(train_meta)), out)
        out = np.concatenate([out, rng.choice(remaining, size=size - len(out), replace=False)])
    return out


def clean_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return arr.astype(np.float64, copy=False)


def save_shap_outputs(
    model_name: str,
    shap_values: np.ndarray,
    shap_x: np.ndarray,
    shap_meta: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    model_dir = OUTDIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    combined = np.mean(np.abs(shap_values), axis=0)
    charge_mask = shap_meta["event_type"].eq("charge").to_numpy()
    discharge_mask = shap_meta["event_type"].eq("discharge").to_numpy()
    charge = np.mean(np.abs(shap_values[charge_mask]), axis=0)
    discharge = np.mean(np.abs(shap_values[discharge_mask]), axis=0)
    for idx, feature in enumerate(feature_names):
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap_combined": float(combined[idx]),
                "mean_abs_shap_charge": float(charge[idx]),
                "mean_abs_shap_discharge": float(discharge[idx]),
            }
        )
    importance = pd.DataFrame(rows)
    for col in ["mean_abs_shap_combined", "mean_abs_shap_charge", "mean_abs_shap_discharge"]:
        importance[f"rank_{col.replace('mean_abs_shap_', '')}"] = importance[col].rank(ascending=False, method="min").astype(int)
    importance = importance.sort_values("mean_abs_shap_combined", ascending=False).reset_index(drop=True)
    importance.to_csv(model_dir / "global_shap_feature_importance.csv", index=False)

    top = importance.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(top))
    ax.barh(y - 0.22, top["mean_abs_shap_charge"], height=0.22, label="charge", color="#2a9d8f")
    ax.barh(y, top["mean_abs_shap_discharge"], height=0.22, label="discharge", color="#e76f51")
    ax.barh(y + 0.22, top["mean_abs_shap_combined"], height=0.22, label="combined", color="#457b9d")
    ax.set_yticks(y, top["feature"])
    ax.set_xlabel("Mean |SHAP| in SOH percentage points")
    ax.set_title(f"{model_name}: global SHAP by event type")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(model_dir / "global_shap_bar_by_event_type.png", dpi=180)
    plt.close(fig)

    for label, mask in [("combined", np.ones(len(shap_x), dtype=bool)), ("charge", charge_mask), ("discharge", discharge_mask)]:
        plt.figure(figsize=(9, 7))
        shap.summary_plot(
            shap_values[mask],
            shap_x[mask],
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.title(f"{model_name}: SHAP beeswarm ({label})")
        plt.tight_layout()
        plt.savefig(model_dir / f"global_shap_beeswarm_{label}.png", dpi=180, bbox_inches="tight")
        plt.close()
    return importance


def run_transformer_shap(
    model: nn.Module,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    bg_idx = background_indices(train_meta, BACKGROUND_SIZE)
    shap_idx = stratified_sample_indices(test_meta, SHAP_SAMPLES_PER_EVENT_TYPE)
    background = torch.tensor(train_x[bg_idx], dtype=torch.float32)
    explain_x = torch.tensor(test_x[shap_idx], dtype=torch.float32)
    explainer = shap.GradientExplainer(model, background)
    values = clean_shap_values(explainer.shap_values(explain_x))
    shap_meta = test_meta.iloc[shap_idx].reset_index(drop=True)
    importance = save_shap_outputs(
        "feature_token_transformer",
        values,
        test_x[shap_idx],
        shap_meta,
        feature_names,
    )
    return importance, shap_idx, shap_meta


def run_lightgbm_shap(
    model: lgb.LGBMRegressor,
    test_raw: np.ndarray,
    test_meta: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    shap_idx = stratified_sample_indices(test_meta, SHAP_SAMPLES_PER_EVENT_TYPE)
    explainer = shap.TreeExplainer(model)
    values = clean_shap_values(explainer.shap_values(test_raw[shap_idx]))
    shap_meta = test_meta.iloc[shap_idx].reset_index(drop=True)
    importance = save_shap_outputs(
        "lightgbm",
        values,
        test_raw[shap_idx],
        shap_meta,
        feature_names,
    )
    return importance, shap_idx, shap_meta


def run_lime_for_model(
    model_name: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    test_meta: pd.DataFrame,
    test_y: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    local_indices: dict[str, int],
    predict_fn: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    model_dir = OUTDIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_x,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=RANDOM_SEED,
    )
    rows = []
    for event_type, idx in local_indices.items():
        exp = explainer.explain_instance(
            test_x[idx],
            predict_fn,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(9, 6)
        plt.title(f"{model_name}: local LIME for {event_type} event")
        plt.tight_layout()
        fig.savefig(model_dir / f"local_lime_{event_type}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        exp.save_to_file(str(model_dir / f"local_lime_{event_type}.html"))
        row = test_meta.iloc[idx]
        for rank, (feature_rule, contribution) in enumerate(exp.as_list(), start=1):
            rows.append(
                {
                    "model": model_name,
                    "event_type": event_type,
                    "rank": rank,
                    "feature_rule": feature_rule,
                    "lime_contribution_soh_pp": float(contribution),
                    "battery": row["battery"],
                    "event_order": int(row["event_order"]),
                    "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
                    "actual_soh_percent": float(test_y[idx]),
                    "predicted_soh_percent": float(y_pred[idx]),
                    "absolute_error_pp": float(abs(y_pred[idx] - test_y[idx])),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(model_dir / "local_lime_explanations.csv", index=False)
    return out


def save_cross_model_summary(
    transformer_importance: pd.DataFrame,
    lightgbm_importance: pd.DataFrame,
    transformer_lime: pd.DataFrame,
    lightgbm_lime: pd.DataFrame,
) -> None:
    merged = transformer_importance[["feature", "mean_abs_shap_combined", "mean_abs_shap_charge", "mean_abs_shap_discharge"]].rename(
        columns={
            "mean_abs_shap_combined": "transformer_mean_abs_shap_combined",
            "mean_abs_shap_charge": "transformer_mean_abs_shap_charge",
            "mean_abs_shap_discharge": "transformer_mean_abs_shap_discharge",
        }
    )
    merged = merged.merge(
        lightgbm_importance[["feature", "mean_abs_shap_combined", "mean_abs_shap_charge", "mean_abs_shap_discharge"]].rename(
            columns={
                "mean_abs_shap_combined": "lightgbm_mean_abs_shap_combined",
                "mean_abs_shap_charge": "lightgbm_mean_abs_shap_charge",
                "mean_abs_shap_discharge": "lightgbm_mean_abs_shap_discharge",
            }
        ),
        on="feature",
        how="outer",
    ).fillna(0.0)
    merged["transformer_rank"] = merged["transformer_mean_abs_shap_combined"].rank(ascending=False, method="min").astype(int)
    merged["lightgbm_rank"] = merged["lightgbm_mean_abs_shap_combined"].rank(ascending=False, method="min").astype(int)
    merged = merged.sort_values("transformer_mean_abs_shap_combined", ascending=False)
    merged.to_csv(OUTDIR / "cross_model_global_shap_comparison.csv", index=False)
    pd.concat([transformer_lime, lightgbm_lime], ignore_index=True).to_csv(OUTDIR / "local_lime_all_models.csv", index=False)

    top_transformer = transformer_importance.head(10)["feature"].tolist()
    top_lightgbm = lightgbm_importance.head(10)["feature"].tolist()
    summary = {
        "shap_background_train_samples": BACKGROUND_SIZE,
        "shap_test_samples_per_event_type": SHAP_SAMPLES_PER_EVENT_TYPE,
        "lime_num_samples": LIME_NUM_SAMPLES,
        "lime_num_features": LIME_NUM_FEATURES,
        "top10_transformer_global_shap": top_transformer,
        "top10_lightgbm_global_shap": top_lightgbm,
    }
    (OUTDIR / "interpretability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    torch.set_num_threads(8)
    data = prepare_data()
    feature_names = data["feature_names"]  # type: ignore[assignment]
    train_meta = data["train_meta"]  # type: ignore[assignment]
    test_meta = data["test_meta"]  # type: ignore[assignment]
    train_x = data["train_x"]  # type: ignore[assignment]
    test_x = data["test_x"]  # type: ignore[assignment]
    train_raw = data["train_raw"]  # type: ignore[assignment]
    test_raw = data["test_raw"]  # type: ignore[assignment]
    train_y = data["train_y"]  # type: ignore[assignment]
    test_y = data["test_y"]  # type: ignore[assignment]
    train_w = data["train_w"]  # type: ignore[assignment]
    y_mean = float(data["y_mean"])
    y_std = float(data["y_std"])

    transformer = load_transformer(len(feature_names), y_mean, y_std)
    transformer_fn = transformer_predict_fn(transformer)
    transformer_pred = transformer_fn(test_x)
    local_indices = choose_indices(test_meta, test_y, transformer_pred)

    lightgbm_model = train_lightgbm(train_raw, train_y, train_w)
    lightgbm_pred = lightgbm_model.predict(test_raw)

    transformer_importance, _, _ = run_transformer_shap(
        transformer,
        train_x,
        test_x,
        train_meta,
        test_meta,
        feature_names,
    )
    lightgbm_importance, _, _ = run_lightgbm_shap(
        lightgbm_model,
        test_raw,
        test_meta,
        feature_names,
    )
    transformer_lime = run_lime_for_model(
        "feature_token_transformer",
        train_x,
        test_x,
        test_meta,
        test_y,
        transformer_pred,
        feature_names,
        local_indices,
        transformer_fn,
    )
    lightgbm_lime = run_lime_for_model(
        "lightgbm",
        train_raw,
        test_raw,
        test_meta,
        test_y,
        lightgbm_pred,
        feature_names,
        local_indices,
        lightgbm_model.predict,
    )
    save_cross_model_summary(transformer_importance, lightgbm_importance, transformer_lime, lightgbm_lime)

    local_instance_summary = []
    for event_type, idx in local_indices.items():
        row = test_meta.iloc[idx]
        local_instance_summary.append(
            {
                "event_type": event_type,
                "test_row_index": int(idx),
                "battery": row["battery"],
                "event_order": int(row["event_order"]),
                "assigned_checkpoint_cycle": int(row["assigned_checkpoint_cycle"]),
                "actual_soh_percent": float(test_y[idx]),
                "transformer_prediction": float(transformer_pred[idx]),
                "lightgbm_prediction": float(lightgbm_pred[idx]),
            }
        )
    pd.DataFrame(local_instance_summary).to_csv(OUTDIR / "local_instances_used.csv", index=False)
    print("Saved SHAP/LIME outputs to", OUTDIR, flush=True)
    print("Top transformer SHAP features:", transformer_importance.head(8)["feature"].tolist(), flush=True)
    print("Top LightGBM SHAP features:", lightgbm_importance.head(8)["feature"].tolist(), flush=True)


if __name__ == "__main__":
    main()
