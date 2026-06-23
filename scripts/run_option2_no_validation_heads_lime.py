from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lime.lime_tabular import LimeTabularExplainer
from torch.utils.data import DataLoader

from run_clean_benchmark_soh_transformers import (
    BenchmarkTransformer,
    ModelSpec,
    SequenceDataset,
    build_dataset_for_batteries,
    destandardize,
    regression_metrics,
    run_epoch,
    sanitize_and_standardize,
    set_seed,
    standardize_target,
)


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = BASE / "option2_no_validation_heads"
TRAIN_BATTERIES = ("RW9", "RW10", "RW11")
TEST_BATTERY = ("RW12",)

TOKEN_FEATURES = [
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

SPECS = [
    ModelSpec("option2_h4", "option2", 64, 4, 2, 128, 0.10, 5.0e-4, 80, 0, 32, segment_history=32),
    ModelSpec("option2_h8", "option2", 64, 8, 2, 128, 0.10, 5.0e-4, 80, 0, 32, segment_history=32),
    ModelSpec("option2_h10", "option2", 80, 10, 2, 160, 0.10, 5.0e-4, 80, 0, 32, segment_history=32),
    ModelSpec("option2_h12", "option2", 96, 12, 2, 192, 0.10, 5.0e-4, 80, 0, 32, segment_history=32),
]


def flatten_feature_names(token_count: int) -> list[str]:
    return [
        f"seg_{segment_idx + 1:02d}_{feature_name}"
        for segment_idx in range(token_count)
        for feature_name in TOKEN_FEATURES
    ]


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


def save_prediction_plots(pred_df: pd.DataFrame, outdir: Path) -> None:
    y_true = pred_df["actual_soh_percent"].to_numpy(dtype=np.float64)
    y_pred = pred_df["predicted_soh_percent"].to_numpy(dtype=np.float64)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(y_true, y_pred, s=24, alpha=0.65)
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[0].set_title("RW12 predicted vs actual SOH")
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")

    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, s=24, alpha=0.65)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("RW12 residuals")
    axes[1].set_xlabel("Actual SOH (%)")
    axes[1].set_ylabel("Error (pp)")
    fig.savefig(outdir / "test_predictions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plot_df = pred_df.sort_values("checkpoint_cycle").reset_index(drop=True)
    ax.plot(plot_df["checkpoint_cycle"], plot_df["actual_soh_percent"], marker="o", linewidth=1.8, label="actual")
    ax.plot(plot_df["checkpoint_cycle"], plot_df["predicted_soh_percent"], marker="o", linewidth=1.8, label="predicted")
    ax.set_title("RW12 SOH over benchmark checkpoints")
    ax.set_xlabel("Reference checkpoint charge-cycle count")
    ax.set_ylabel("SOH (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "test_soh_over_checkpoints.png", dpi=180)
    plt.close(fig)


def make_predict_fn(model: BenchmarkTransformer, fixed_mask: np.ndarray, y_mean: float, y_std: float, token_count: int, input_dim: int):
    device = next(model.parameters()).device

    def predict_fn(flat_x: np.ndarray) -> np.ndarray:
        seq = flat_x.reshape(len(flat_x), token_count, input_dim).astype(np.float32)
        mask = np.repeat(fixed_mask[None, :], len(flat_x), axis=0)
        with torch.no_grad():
            pred_norm = model(
                torch.from_numpy(seq).float().to(device),
                torch.from_numpy(mask).bool().to(device),
            ).cpu().numpy()
        return destandardize(pred_norm, y_mean, y_std)

    return predict_fn


def run_global_lime(
    model: BenchmarkTransformer,
    train_x: np.ndarray,
    test_x: np.ndarray,
    test_mask: np.ndarray,
    y_mean: float,
    y_std: float,
    outdir: Path,
    token_count: int,
    input_dim: int,
) -> None:
    flat_train = train_x.reshape(len(train_x), -1)
    flat_test = test_x.reshape(len(test_x), -1)
    feature_names = flatten_feature_names(token_count)
    explainer = LimeTabularExplainer(
        training_data=flat_train,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=False,
        random_state=42,
    )

    feature_scores = {name: 0.0 for name in TOKEN_FEATURES}
    flattened_scores = {name: 0.0 for name in feature_names}

    for idx in range(len(flat_test)):
        predict_fn = make_predict_fn(model, test_mask[idx], y_mean, y_std, token_count, input_dim)
        exp = explainer.explain_instance(
            flat_test[idx],
            predict_fn,
            num_features=min(80, len(feature_names)),
            num_samples=1200,
        )
        local_map = exp.local_exp[list(exp.local_exp.keys())[0]]
        for feat_idx, weight in local_map:
            flat_name = feature_names[feat_idx]
            flattened_scores[flat_name] += abs(weight)
            base_name = flat_name.split("_", 2)[-1]
            feature_scores[base_name] += abs(weight)

    flat_df = (
        pd.DataFrame(
            {"feature": list(flattened_scores.keys()), "global_lime_abs_weight": list(flattened_scores.values())}
        )
        .sort_values("global_lime_abs_weight", ascending=False)
        .reset_index(drop=True)
    )
    base_df = (
        pd.DataFrame(
            {"feature": list(feature_scores.keys()), "global_lime_abs_weight": list(feature_scores.values())}
        )
        .sort_values("global_lime_abs_weight", ascending=False)
        .reset_index(drop=True)
    )
    flat_df.to_csv(outdir / "global_lime_flattened.csv", index=False)
    base_df.to_csv(outdir / "global_lime_features.csv", index=False)

    top = base_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(top["feature"], top["global_lime_abs_weight"], color="#5a189a")
    ax.set_title("Global LIME feature importance")
    ax.set_xlabel("Aggregated absolute LIME weight")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(outdir / "global_lime_features.png", dpi=180)
    plt.close(fig)


def train_and_test(spec: ModelSpec) -> dict[str, object]:
    train_x, train_mask, train_meta = build_dataset_for_batteries(spec, TRAIN_BATTERIES)
    test_x, test_mask, test_meta = build_dataset_for_batteries(spec, TEST_BATTERY)
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
    for epoch in range(1, spec.max_epochs + 1):
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

    train_loss, train_pred_norm, train_true_norm, train_battery_ids, train_cycles, train_ids = run_epoch(
        model, train_loader, None, device
    )
    test_loss, test_pred_norm, test_true_norm, test_battery_ids, test_cycles, test_ids = run_epoch(
        model, test_loader, None, device
    )
    train_pred = destandardize(train_pred_norm, y_mean, y_std)
    train_true = destandardize(train_true_norm, y_mean, y_std)
    test_pred = destandardize(test_pred_norm, y_mean, y_std)
    test_true = destandardize(test_true_norm, y_mean, y_std)

    return {
        "spec": spec,
        "history": pd.DataFrame(history_rows),
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
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "state_dict": model.state_dict(),
        "model": model,
        "train_x": train_x,
        "test_x": test_x,
        "test_mask": test_mask,
        "input_dim": train_x.shape[-1],
    }


def save_run_artifacts(result: dict[str, object], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    result["history"].to_csv(outdir / "training_history.csv", index=False)
    result["train_predictions"].to_csv(outdir / "train_predictions.csv", index=False)
    result["test_predictions"].to_csv(outdir / "test_predictions.csv", index=False)
    save_loss_plot(result["history"], outdir, f"{result['spec'].name}: train/test loss")
    save_prediction_plots(result["test_predictions"], outdir)
    torch.save(result["state_dict"], outdir / "model.pt")
    summary = {
        "spec": result["spec"].__dict__,
        "parameter_count": result["parameter_count"],
        "train_metrics": result["train_metrics"],
        "test_metrics": result["test_metrics"],
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    rows: list[dict[str, object]] = []
    best_result = None
    best_test_r2 = -np.inf

    for spec in SPECS:
        print(f"Running {spec.name} ...", flush=True)
        result = train_and_test(spec)
        run_dir = OUTDIR / spec.name
        save_run_artifacts(result, run_dir)
        rows.append(
            {
                "model": spec.name,
                "heads": spec.nhead,
                "layers": spec.num_layers,
                "d_model": spec.d_model,
                "token_count": spec.token_count,
                "train_mae_pp": result["train_metrics"]["mae"],
                "train_rmse_pp": result["train_metrics"]["rmse"],
                "train_r2": result["train_metrics"]["r2"],
                "test_mae_pp": result["test_metrics"]["mae"],
                "test_rmse_pp": result["test_metrics"]["rmse"],
                "test_r2": result["test_metrics"]["r2"],
                "test_mape_percent": result["test_metrics"]["mape_percent"],
            }
        )
        if result["test_metrics"]["r2"] > best_test_r2:
            best_test_r2 = result["test_metrics"]["r2"]
            best_result = result

    summary_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    summary_df.to_csv(OUTDIR / "heads_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(summary_df["model"], summary_df["test_r2"], color="#1d3557")
    axes[0].set_title("RW12 test R² by head count")
    axes[0].set_ylabel("R²")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(summary_df["model"], summary_df["test_mae_pp"], color="#e76f51")
    axes[1].set_title("RW12 test MAE by head count")
    axes[1].set_ylabel("MAE (SOH pp)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUTDIR / "heads_comparison.png", dpi=180)
    plt.close(fig)

    if best_result is None:
        raise RuntimeError("No runs completed.")

    best_dir = OUTDIR / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    run_global_lime(
        model=best_result["model"],
        train_x=best_result["train_x"],
        test_x=best_result["test_x"],
        test_mask=best_result["test_mask"],
        y_mean=best_result["target_mean"],
        y_std=best_result["target_std"],
        outdir=best_dir,
        token_count=best_result["spec"].token_count,
        input_dim=best_result["input_dim"],
    )
    save_run_artifacts(best_result, best_dir)
    (best_dir / "best_model_name.txt").write_text(best_result["spec"].name, encoding="utf-8")

    report_lines = [
        "# Option 2 No-Validation Head Search",
        "",
        "Sample definition:",
        "- One sample = one benchmark SOH target point",
        "- Tokens = last 32 partial charge segments before that benchmark",
        "- One token = one segment-level engineered feature vector",
        "",
        "Train/Test split:",
        "- Train: full RW9 + RW10 + RW11",
        "- Test: full RW12",
        "",
        "Results:",
    ]
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"- {row['model']}: test R2={row['test_r2']:.4f}, "
            f"test MAE={row['test_mae_pp']:.4f} pp, test RMSE={row['test_rmse_pp']:.4f} pp"
        )
    (OUTDIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
