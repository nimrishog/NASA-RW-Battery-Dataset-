from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "reviewer_validation" / "stage7_architecture_dq_dv" / "four_condition_metrics.csv"
OUT = ROOT / "results" / "reviewer_validation" / "stage8_architecture_feature_interaction"
METRICS = ("event_rmse", "event_mae", "event_r2", "checkpoint_rmse", "checkpoint_mae", "checkpoint_r2")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    rows = []
    for architecture in ("transformer", "rnn"):
        subset = data[data["architecture"].eq(architecture)].pivot(index="seed", columns="condition")
        for seed in sorted(data["seed"].unique()):
            row = {"seed": seed, "architecture": architecture}
            for metric in METRICS:
                row[f"delta_{metric}"] = subset[metric].loc[seed, "dq_dv_removed"] - subset[metric].loc[seed, "full"]
            rows.append(row)
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT / "within_architecture_paired_effects.csv", index=False)

    wide = effects.pivot(index="seed", columns="architecture")
    interactions = pd.DataFrame({"seed": sorted(data["seed"].unique())})
    for metric in METRICS:
        interactions[f"interaction_{metric}"] = wide[f"delta_{metric}"]["rnn"].to_numpy() - wide[f"delta_{metric}"]["transformer"].to_numpy()
    interactions.to_csv(OUT / "seed_specific_interactions.csv", index=False)

    rows = []
    for column in interactions.columns[1:]:
        values = interactions[column].to_numpy()
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        se = sd / np.sqrt(len(values))
        critical = float(t.ppf(0.975, df=len(values) - 1))
        rows.append({"interaction": column, "mean": mean, "sd": sd, "ci95_low": mean - critical * se, "ci95_high": mean + critical * se, "n_seeds": len(values)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "interaction_summary.csv", index=False)
    print("WITHIN-ARCHITECTURE EFFECTS (removed minus full)\n", effects.to_string(index=False), flush=True)
    print("\nINTERACTIONS (RNN effect minus Transformer effect)\n", interactions.to_string(index=False), flush=True)
    print("\nSUMMARY\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
