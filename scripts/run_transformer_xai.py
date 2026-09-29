"""Generate the Gradient SHAP and LIME outputs reported in the manuscript."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.linear_model import Ridge

import run_transformer_loco as pipeline


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "primary_transformer" / "model.pt"
OUT = ROOT / "results" / "primary_transformer" / "xai"
PERTURBATIONS = 700
KERNEL_WIDTH = 2.0
RIDGE_ALPHA = 0.3
SEED = 42


def windows(frame: pd.DataFrame, length: int) -> tuple[np.ndarray, pd.DataFrame]:
    arrays = []
    metadata = []
    for battery, block in frame.groupby("battery", sort=False):
        block = block.sort_values("event_order_chronological").reset_index(drop=True)
        values = block[pipeline.FEATURES].to_numpy(np.float32)
        view = np.lib.stride_tricks.sliding_window_view(values, (length, values.shape[1]))[:, 0]
        arrays.append(view)
        metadata.append(block.iloc[length - 1 :].reset_index(drop=True))
    return np.concatenate(arrays), pd.concat(metadata, ignore_index=True)


def load_model_and_data():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = pipeline.EventTransformer(len(checkpoint["features"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    events = pipeline.load_events()
    raw, meta = windows(events, config["window"])
    scaler = checkpoint["scaler"]
    normalized = (np.clip(raw, scaler["lo"], scaler["hi"]) - scaler["mean"]) / scaler["std"]
    return model, normalized.astype(np.float32), meta, scaler


def predict(model, values, scaler):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            prediction = model(torch.from_numpy(values[start : start + 2048])).numpy()
            outputs.append(prediction * scaler["y_std"] + scaler["y_mean"])
    return np.concatenate(outputs)


def gradient_shap(model, values, meta, scaler):
    train_mask = meta["battery"].isin(["RW9", "RW10"]).to_numpy()
    test_mask = meta["battery"].eq("RW11").to_numpy()
    rng = np.random.default_rng(SEED)
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    background_indices = rng.choice(train_indices, size=min(128, len(train_indices)), replace=False)
    explained_indices = np.linspace(0, len(test_indices) - 1, min(512, len(test_indices)), dtype=int)
    background = torch.from_numpy(values[background_indices])
    explained = torch.from_numpy(values[test_indices[explained_indices]])
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(explained)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    attribution = np.asarray(shap_values) * scaler["y_std"]
    importance = np.mean(np.abs(attribution), axis=(0, 1))
    table = pd.DataFrame({"feature": pipeline.FEATURES, "mean_abs_gradient_shap": importance})
    table = table.sort_values("mean_abs_gradient_shap", ascending=False)
    table.to_csv(OUT / "global_gradient_shap.csv", index=False)
    plot = table.head(15).sort_values("mean_abs_gradient_shap")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(plot["feature"], plot["mean_abs_gradient_shap"], color="#2f80ed")
    axis.set_xlabel("Mean absolute Gradient SHAP attribution (SOH points)")
    fig.tight_layout()
    fig.savefig(OUT / "global_gradient_shap.png", dpi=300)
    plt.close(fig)


def lime(model, values, meta, scaler):
    test_mask = meta["battery"].eq("RW11").to_numpy()
    test_values = values[test_mask]
    test_meta = meta.loc[test_mask].reset_index(drop=True)
    predictions = predict(model, test_values, scaler)
    errors = np.abs(predictions - test_meta["soh_percent"].to_numpy())
    rng = np.random.default_rng(SEED)
    rows = []
    for event_type in ("charge", "discharge"):
        candidates = np.flatnonzero(test_meta["event_type"].eq(event_type).to_numpy())
        selected = candidates[np.argsort(errors[candidates])[len(candidates) // 2]]
        instance = test_values[selected]
        noise = rng.normal(size=(PERTURBATIONS, *instance.shape)).astype(np.float32)
        perturbed = instance[None, :, :] + noise
        distances = np.sqrt(np.mean(noise**2, axis=(1, 2)))
        weights = np.exp(-(distances**2) / (KERNEL_WIDTH**2))
        responses = predict(model, perturbed, scaler)
        summary = perturbed.mean(axis=1)
        surrogate = Ridge(alpha=RIDGE_ALPHA).fit(summary, responses, sample_weight=weights)
        for feature, coefficient in zip(pipeline.FEATURES, surrogate.coef_):
            rows.append({"event_type": event_type, "feature": feature, "coefficient": coefficient})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "local_lime.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    for axis, event_type in zip(axes, ("charge", "discharge")):
        plot = table[table["event_type"].eq(event_type)].copy()
        plot["magnitude"] = plot["coefficient"].abs()
        plot = plot.nlargest(10, "magnitude").sort_values("coefficient")
        axis.barh(plot["feature"], plot["coefficient"], color=np.where(plot["coefficient"] >= 0, "#2f80ed", "#d95f59"))
        axis.set_title(event_type.capitalize())
    fig.tight_layout()
    fig.savefig(OUT / "local_lime.png", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, values, meta, scaler = load_model_and_data()
    gradient_shap(model, values, meta, scaler)
    lime(model, values, meta, scaler)
    (OUT / "settings.json").write_text(
        json.dumps({"perturbations": PERTURBATIONS, "kernel_width": KERNEL_WIDTH, "ridge_alpha": RIDGE_ALPHA}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
