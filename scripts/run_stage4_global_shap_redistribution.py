from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

import run_transformer_loco as pipeline


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "results" / "reviewer_validation" / "stage2_paired_dq_dv_seeds" / "seed_42"
OUT = ROOT / "results" / "reviewer_validation" / "stage4_global_shap_redistribution"
MODEL_PATHS = {
    "full": STAGE2 / "full" / "test_RW11" / "best_model.pt",
    "dq_dv_removed": STAGE2 / "dq_dv_removed" / "test_RW11" / "best_model.pt",
}
SEED = 42
BACKGROUND_SIZE = 128
EXPLAINED_SIZE = 512


class ShapModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.model(values).unsqueeze(1)


def explain(condition: str, checkpoint: dict, events: pd.DataFrame, features: list[str], background_indices: np.ndarray, explained_indices: np.ndarray) -> pd.DataFrame:
    pipeline.FEATURES = features
    split = pipeline.prepare_split(events, "RW11")
    model = pipeline.EventTransformer(len(features))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    background = torch.from_numpy(split["x_train"][background_indices])
    explained = torch.from_numpy(split["x_test"][explained_indices])
    explainer = shap.GradientExplainer(ShapModel(model), background)
    attribution = explainer.shap_values(explained)
    if isinstance(attribution, list):
        attribution = attribution[0]
    attribution = np.asarray(attribution)
    if attribution.ndim == 4 and attribution.shape[-1] == 1:
        attribution = attribution[..., 0]
    attribution = attribution * split["scaler"]["y_std"]
    importance = np.mean(np.abs(attribution), axis=(0, 1))
    signed = np.mean(attribution, axis=(0, 1))
    table = pd.DataFrame({"condition": condition, "feature": features, "mean_abs_shap": importance, "mean_signed_shap": signed})
    table["importance_share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    table["rank"] = table["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    return table


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in MODEL_PATHS.items()}
    full_features = list(pipeline.FEATURES)
    feature_sets = {"full": full_features, "dq_dv_removed": [name for name in full_features if name != "dq_dv_max_ahpv"]}
    pipeline.FEATURES = full_features
    events = pipeline.load_events()
    split = pipeline.prepare_split(events, "RW11")
    rng = np.random.default_rng(SEED)
    background_indices = rng.choice(len(split["x_train"]), size=BACKGROUND_SIZE, replace=False)
    explained_indices = np.linspace(0, len(split["x_test"]) - 1, EXPLAINED_SIZE, dtype=int)

    tables = [explain(name, checkpoint, events, feature_sets[name], background_indices, explained_indices) for name, checkpoint in checkpoints.items()]
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUT / "global_gradient_shap_by_condition.csv", index=False)
    wide = combined.pivot(index="feature", columns="condition", values=["mean_abs_shap", "importance_share", "rank"])
    common = wide.dropna().copy()
    comparison = pd.DataFrame(index=common.index)
    comparison["full_mean_abs_shap"] = common["mean_abs_shap"]["full"]
    comparison["ablated_mean_abs_shap"] = common["mean_abs_shap"]["dq_dv_removed"]
    comparison["delta_mean_abs_shap"] = comparison["ablated_mean_abs_shap"] - comparison["full_mean_abs_shap"]
    comparison["full_share"] = common["importance_share"]["full"]
    comparison["ablated_share"] = common["importance_share"]["dq_dv_removed"]
    comparison["delta_share"] = comparison["ablated_share"] - comparison["full_share"]
    comparison["full_rank"] = common["rank"]["full"].astype(int)
    comparison["ablated_rank"] = common["rank"]["dq_dv_removed"].astype(int)
    comparison = comparison.sort_values("delta_share", ascending=False)
    comparison.to_csv(OUT / "global_shap_redistribution_common_features.csv")
    totals = combined.groupby("condition")["mean_abs_shap"].sum().rename("total_mean_abs_shap").reset_index()
    totals.to_csv(OUT / "global_shap_total_attribution.csv", index=False)
    print("TOP POSITIVE SHARE CHANGES\n", comparison.head(10).to_string(), flush=True)
    print("\nTOTALS\n", totals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
