from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import run_transformer_loco as pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_validation" / "stage2_paired_dq_dv_seeds"
SEEDS = (42, 43, 44)
FULL_FEATURES = list(pipeline.FEATURES)
ABLATION_FEATURES = [name for name in FULL_FEATURES if name != "dq_dv_max_ahpv"]


def train_condition(events: pd.DataFrame, seed: int, condition: str, features: list[str]) -> dict[str, object]:
    run_dir = OUT / f"seed_{seed}" / condition
    metrics_path = run_dir / "test_RW11" / "metrics.json"
    pipeline.CFG = replace(pipeline.CFG, seed=seed)
    pipeline.FEATURES = features
    pipeline.OUT = run_dir
    if metrics_path.exists():
        row = json.loads(metrics_path.read_text(encoding="utf-8"))["row"]
        print(f"Using completed seed={seed} condition={condition}", flush=True)
    else:
        print(f"Training seed={seed} condition={condition}", flush=True)
        row = pipeline.train_one_loco(events, "RW11")
    return {"seed": seed, "condition": condition, **row}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline.FEATURES = FULL_FEATURES
    events = pipeline.load_events()
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rows.append(train_condition(events, seed, "full", FULL_FEATURES))
        rows.append(train_condition(events, seed, "dq_dv_removed", ABLATION_FEATURES))
        pd.DataFrame(rows).to_csv(OUT / "paired_seed_metrics_partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "paired_seed_metrics.csv", index=False)
    wide = metrics.pivot(index="seed", columns="condition")
    effects = pd.DataFrame({"seed": list(SEEDS)})
    for metric in ("event_rmse", "event_mae", "event_r2", "checkpoint_rmse", "checkpoint_mae", "checkpoint_r2"):
        effects[f"delta_{metric}"] = wide[metric]["dq_dv_removed"].to_numpy() - wide[metric]["full"].to_numpy()
    effects.to_csv(OUT / "paired_seed_effects.csv", index=False)
    summary = effects.drop(columns="seed").agg(["mean", "std", "min", "max"]).T.reset_index(names="effect")
    summary.to_csv(OUT / "paired_seed_effect_summary.csv", index=False)
    print("\nPAIRED EFFECTS (removed minus full)\n", effects.to_string(index=False), flush=True)
    print("\nSUMMARY\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
