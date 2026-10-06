"""Fast integrity checks for the publication repository and reported outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PRIMARY = {
    "test_event": {"r2": 0.9550821781, "mae": 2.3866641521, "rmse": 3.0388083070},
    "test_checkpoint": {"r2": 0.9663872719, "mae": 2.0689485073, "rmse": 2.5350211038},
}
EXPECTED_BASELINE_RMSE = {
    "charge_count_ridge": (6.3239388179, 6.3991515883),
    "elapsed_time_ridge": (6.4443129346, 6.5508148138),
    "throughput_ridge": (5.8497431669, 6.0601048653),
}
EXPECTED_FIGURES = {
    "fig_event_transformer_workflow.png",
    "fig_dataset_counts_rw9_rw11.png",
    "fig_capacity_soh_rw9_rw11.png",
    "fig_current_voltage_ranges_rw9_rw11.png",
    "fig_engineered_feature_behavior_rw9_rw11.png",
    "fig_label_strategy_trajectories.png",
    "fig_label_strategy_checkpoint_rmse.png",
    "fig_controlled_neural_baseline_rmse.png",
    "fig_full_transformer_loco_rmse.png",
    "fig_window_length_rmse.png",
    "fig_feature_ablation_rmse.png",
    "fig_dq_dv_ablation_seeds_cells.png",
    "fig_dq_dv_training_dependence.png",
    "fig_event_transformer_pred_vs_actual.png",
    "fig_event_transformer_soh_trajectory.png",
    "fig_xai_global_event_transformer.png",
    "fig_xai_local_lime_charge_discharge.png",
    "fig_dq_dv_shap_redistribution.png",
}


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"Expected {expected}, found {actual}")


def main() -> None:
    primary = json.loads((ROOT / "results" / "primary_transformer" / "metrics.json").read_text(encoding="utf-8"))
    for scale, values in EXPECTED_PRIMARY.items():
        for metric, expected in values.items():
            close(float(primary[scale][metric]), expected)

    baseline_path = ROOT / "results" / "reviewer_validation" / "trend_baselines" / "trend_baseline_metrics.csv"
    if not baseline_path.exists():
        raise FileNotFoundError("Run scripts/run_trend_baselines.py before validation")
    baselines = pd.read_csv(baseline_path).set_index("model")
    for model, (event_rmse, checkpoint_rmse) in EXPECTED_BASELINE_RMSE.items():
        close(float(baselines.loc[model, "event_rmse"]), event_rmse)
        close(float(baselines.loc[model, "checkpoint_rmse"]), checkpoint_rmse)

    figure_dir = ROOT / "results" / "manuscript_figures"
    missing = sorted(name for name in EXPECTED_FIGURES if not (figure_dir / name).exists())
    if missing:
        raise FileNotFoundError(f"Missing manuscript figures: {missing}")

    event_table = ROOT / "data" / "modeling" / "event_features_rw9_rw11.csv"
    columns = pd.read_csv(event_table, nrows=1).columns
    required = {"battery", "event_order_chronological", "soh_percent", "dq_dv_max_ahpv"}
    absent = sorted(required.difference(columns))
    if absent:
        raise ValueError(f"Missing event-table columns: {absent}")

    print("Repository validation passed")
    print("Primary metrics, ridge baselines, event schema, and 18 manuscript figures are present and consistent.")


if __name__ == "__main__":
    main()
