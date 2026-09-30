from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

import run_transformer_loco as pipeline


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "results" / "reviewer_validation" / "stage2_paired_dq_dv_seeds" / "seed_42"
OUT = ROOT / "results" / "reviewer_validation" / "stage5_matched_lime_redistribution"
MODEL_PATHS = {
    "full": STAGE2 / "full" / "test_RW11" / "best_model.pt",
    "dq_dv_removed": STAGE2 / "dq_dv_removed" / "test_RW11" / "best_model.pt",
}
SEED = 42
PERTURBATIONS = 700
KERNEL_WIDTH = 2.0
RIDGE_ALPHA = 0.3


def predict(model, values: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(values.astype(np.float32))).numpy() * y_std + y_mean


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_features = list(pipeline.FEATURES)
    feature_sets = {"full": full_features, "dq_dv_removed": [name for name in full_features if name != "dq_dv_max_ahpv"]}
    pipeline.FEATURES = full_features
    events = pipeline.load_events()
    full_split = pipeline.prepare_split(events, "RW11")
    metadata = full_split["meta_test"].reset_index(drop=True)

    selected = {}
    for event_type in ("charge", "discharge"):
        candidates = np.flatnonzero(metadata["event_type"].eq(event_type).to_numpy())
        selected[event_type] = int(candidates[len(candidates) // 2])
    selection = metadata.iloc[list(selected.values())][["battery", "event_type", "event_order_chronological", "assigned_checkpoint_cycle"]].copy()
    selection.insert(0, "test_window_index", list(selected.values()))
    selection.to_csv(OUT / "predefined_matched_windows.csv", index=False)

    rng = np.random.default_rng(SEED)
    full_noise = rng.normal(size=(PERTURBATIONS, pipeline.CFG.window, len(full_features))).astype(np.float32)
    rows = []
    for condition, features in feature_sets.items():
        pipeline.FEATURES = features
        split = pipeline.prepare_split(events, "RW11")
        checkpoint = torch.load(MODEL_PATHS[condition], map_location="cpu", weights_only=False)
        model = pipeline.EventTransformer(len(features))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        feature_indices = [full_features.index(name) for name in features]
        noise = full_noise[:, :, feature_indices]
        for event_type, index in selected.items():
            instance = split["x_test"][index]
            perturbed = instance[None, :, :] + noise
            distances = np.sqrt(np.mean(noise**2, axis=(1, 2)))
            weights = np.exp(-(distances**2) / (KERNEL_WIDTH**2))
            responses = predict(model, perturbed, split["scaler"]["y_mean"], split["scaler"]["y_std"])
            surrogate = Ridge(alpha=RIDGE_ALPHA).fit(perturbed.mean(axis=1), responses, sample_weight=weights)
            prediction = float(predict(model, instance[None, :, :], split["scaler"]["y_mean"], split["scaler"]["y_std"])[0])
            for feature, coefficient in zip(features, surrogate.coef_):
                rows.append({
                    "condition": condition,
                    "event_type": event_type,
                    "test_window_index": index,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "absolute_coefficient": float(abs(coefficient)),
                    "model_prediction": prediction,
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "matched_local_lime_coefficients.csv", index=False)
    common = table.pivot(index=["event_type", "feature"], columns="condition", values="absolute_coefficient").dropna()
    common["delta_absolute_coefficient"] = common["dq_dv_removed"] - common["full"]
    common.sort_values(["event_type", "delta_absolute_coefficient"], ascending=[True, False]).to_csv(OUT / "matched_lime_redistribution_common_features.csv")
    print(selection.to_string(index=False), flush=True)
    for event_type in ("charge", "discharge"):
        print(f"\n{event_type.upper()} TOP POSITIVE CHANGES\n", common.loc[event_type].nlargest(8, "delta_absolute_coefficient").to_string(), flush=True)


if __name__ == "__main__":
    main()
