from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


BASE = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = BASE / "rw9" / "full_charge_report"
DATE_FMT = "%d-%b-%Y %H:%M:%S"
DATE_ONLY_FMT = "%d-%b-%Y"


def as_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    return np.array([float(value)], dtype=np.float64)


def parse_step_date(raw_value: object) -> pd.Timestamp:
    text = str(raw_value)
    for fmt in (DATE_FMT, DATE_ONLY_FMT):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def robust_stat(values: np.ndarray, fn) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(fn(values))


def load_rw9_charge_report_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pkl = pd.read_pickle(BASE / "rw9_datacapa.pkl")
    pkl_cycle = (
        pkl.groupby("cycle", sort=True)
        .agg(
            pkl_sample_count=("Voltage", "size"),
            capacity_ah=("Capacity", "first"),
            pkl_time_start=("time", "min"),
            pkl_time_end=("time", "max"),
        )
        .reset_index()
    )

    data = scipy.io.loadmat(BASE / "RW9.mat", struct_as_record=False, squeeze_me=True)["data"]
    steps = data.step

    cycle = 0
    cycle_rows: list[dict[str, object]] = []
    sampled_rows: list[dict[str, object]] = []

    for mat_step_index, step in enumerate(steps, start=1):
        comment = str(step.comment)
        step_type = str(step.type)
        if comment != "charge (random walk)" or step_type != "C":
            continue

        time_vec = as_vector(step.time)
        rel_vec = as_vector(step.relativeTime)
        voltage_vec = as_vector(step.voltage)
        current_vec = as_vector(step.current)
        temp_vec = as_vector(step.temperature)
        sample_count = int(time_vec.size)
        if sample_count <= 1:
            continue

        cycle += 1
        date_dt = parse_step_date(step.date)
        duration_s = float(rel_vec[-1] - rel_vec[0])
        dt = np.diff(rel_vec)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        delta_t = np.diff(rel_vec)
        delta_v = np.diff(voltage_vec)
        delta_temp = np.diff(temp_vec)
        dq = -current_vec[:-1] * np.clip(delta_t, 0.0, None) / 3600.0

        dv_dt = np.divide(delta_v, delta_t, out=np.full_like(delta_v, np.nan), where=np.abs(delta_t) > 1.0e-8)
        dtemp_dt = np.divide(
            delta_temp,
            delta_t,
            out=np.full_like(delta_temp, np.nan),
            where=np.abs(delta_t) > 1.0e-8,
        )
        dq_dv = np.divide(dq, delta_v, out=np.full_like(dq, np.nan), where=np.abs(delta_v) > 1.0e-6)
        power_in = voltage_vec * (-current_vec)
        energy_in_wh = float(np.trapz(power_in, rel_vec) / 3600.0)
        charged_ah = float(-np.trapz(current_vec, rel_vec) / 3600.0)

        cycle_rows.append(
            {
                "cycle": cycle,
                "mat_step_index": mat_step_index,
                "date": str(step.date),
                "date_dt": date_dt,
                "sample_count": sample_count,
                "duration_s": duration_s,
                "time_start": float(time_vec[0]),
                "time_end": float(time_vec[-1]),
                "current_mean": float(np.mean(current_vec)),
                "current_abs_mean": float(np.mean(-current_vec)),
                "current_std": float(np.std(current_vec)),
                "voltage_start": float(voltage_vec[0]),
                "voltage_end": float(voltage_vec[-1]),
                "voltage_mean": float(np.mean(voltage_vec)),
                "voltage_delta": float(voltage_vec[-1] - voltage_vec[0]),
                "temperature_start": float(temp_vec[0]),
                "temperature_end": float(temp_vec[-1]),
                "temperature_mean": float(np.mean(temp_vec)),
                "temperature_delta": float(temp_vec[-1] - temp_vec[0]),
                "charged_ah": charged_ah,
                "energy_in_wh": energy_in_wh,
                "power_mean_w": float(np.mean(power_in)),
                "dv_dt_mean_vps": robust_stat(dv_dt, np.mean),
                "dv_dt_median_vps": robust_stat(dv_dt, np.median),
                "dtemp_dt_mean_cps": robust_stat(dtemp_dt, np.mean),
                "dtemp_dt_median_cps": robust_stat(dtemp_dt, np.median),
                "dq_dv_median_ahpv": robust_stat(dq_dv, np.median),
                "dt_median_s": robust_stat(dt, np.median),
            }
        )

        sample_keep = np.unique(np.linspace(0, sample_count - 1, num=min(5, sample_count), dtype=int))
        for idx in sample_keep:
            sampled_rows.append(
                {
                    "cycle": cycle,
                    "date_dt": date_dt,
                    "time": float(time_vec[idx]),
                    "relTime": float(rel_vec[idx]),
                    "Voltage": float(voltage_vec[idx]),
                    "Current": float(current_vec[idx]),
                    "Temperature": float(temp_vec[idx]),
                    "PowerIn": float(power_in[idx]),
                }
            )

    cycle_df = pd.DataFrame(cycle_rows)
    sampled_df = pd.DataFrame(sampled_rows)

    merged = cycle_df.merge(pkl_cycle, on="cycle", how="left", validate="one_to_one")
    if len(merged) != len(cycle_df):
        raise ValueError("Cycle merge with pkl labels failed.")

    merged["sample_count_match"] = merged["sample_count"].eq(merged["pkl_sample_count"])
    merged["time_start_abs_diff_s"] = np.abs(merged["time_start"] - merged["pkl_time_start"])
    if not merged["sample_count_match"].all():
        bad = merged.loc[~merged["sample_count_match"], "cycle"].tolist()[:10]
        raise ValueError(f"MAT/pkl sample-count mismatch for cycles: {bad}")

    merged["elapsed_days"] = (merged["time_start"] - merged["time_start"].min()) / 86400.0
    sampled_df["elapsed_days"] = (sampled_df["time"] - merged["time_start"].min()) / 86400.0

    sampled_df = sampled_df.merge(merged[["cycle", "capacity_ah"]], on="cycle", how="left", validate="many_to_one")
    return merged, sampled_df


def save_outputs(cycle_df: pd.DataFrame, sampled_df: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cycle_df.to_csv(OUTDIR / "rw9_charge_cycle_features_full_report.csv", index=False)
    sampled_df.to_csv(OUTDIR / "rw9_charge_overview_samples.csv", index=False)


def plot_voltage_temperature(sampled_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].scatter(sampled_df["elapsed_days"], sampled_df["Voltage"], s=8, alpha=0.55, color="#1f77b4")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("RW9 charging data: voltage and temperature over time")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(sampled_df["elapsed_days"], sampled_df["Temperature"], s=8, alpha=0.55, color="#d1495b")
    axes[1].set_ylabel("Temperature (C)")
    axes[1].set_xlabel("Elapsed days from first exported charge cycle")
    axes[1].grid(alpha=0.25)
    fig.savefig(OUTDIR / "01_voltage_temperature_over_time.png", dpi=180)
    plt.close(fig)


def plot_current_power(sampled_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].scatter(sampled_df["elapsed_days"], -sampled_df["Current"], s=8, alpha=0.55, color="#00798c")
    axes[0].set_ylabel("Charge current magnitude (A)")
    axes[0].set_title("RW9 charging data: current and power over time")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(sampled_df["elapsed_days"], sampled_df["PowerIn"], s=8, alpha=0.55, color="#edae49")
    axes[1].set_ylabel("Power in (W)")
    axes[1].set_xlabel("Elapsed days from first exported charge cycle")
    axes[1].grid(alpha=0.25)
    fig.savefig(OUTDIR / "02_current_power_over_time.png", dpi=180)
    plt.close(fig)


def plot_derivatives(cycle_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    x = cycle_df["elapsed_days"]

    axes[0, 0].plot(x, cycle_df["charged_ah"], linewidth=1.0, color="#1f77b4")
    axes[0, 0].set_title("Charged Ah per cycle")
    axes[0, 0].set_ylabel("Ah")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(x, cycle_df["energy_in_wh"], linewidth=1.0, color="#00798c")
    axes[0, 1].set_title("Energy in per cycle")
    axes[0, 1].set_ylabel("Wh")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(x, cycle_df["dv_dt_median_vps"], linewidth=1.0, color="#d1495b")
    axes[1, 0].set_title("Median dV/dt per cycle")
    axes[1, 0].set_ylabel("V/s")
    axes[1, 0].set_xlabel("Elapsed days")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(x, cycle_df["dq_dv_median_ahpv"], linewidth=1.0, color="#6c757d")
    axes[1, 1].set_title("Median dQ/dV per cycle")
    axes[1, 1].set_ylabel("Ah/V")
    axes[1, 1].set_xlabel("Elapsed days")
    axes[1, 1].grid(alpha=0.25)

    fig.savefig(OUTDIR / "03_charge_related_features_and_derivatives.png", dpi=180)
    plt.close(fig)


def plot_capacity_over_time(cycle_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].step(cycle_df["elapsed_days"], cycle_df["capacity_ah"], where="post", color="#2a9d8f", linewidth=1.8)
    axes[0].set_title("RW9 pkl capacity label over time")
    axes[0].set_ylabel("Capacity label (Ah)")
    axes[0].grid(alpha=0.25)

    axes[1].step(cycle_df["date_dt"], cycle_df["capacity_ah"], where="post", color="#e76f51", linewidth=1.8)
    axes[1].set_ylabel("Capacity label (Ah)")
    axes[1].set_xlabel("Calendar date")
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))
    fig.savefig(OUTDIR / "04_capacity_over_time.png", dpi=180)
    plt.close(fig)


def plot_capacity_vs_features(cycle_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    y = cycle_df["capacity_ah"]
    plots = [
        ("charged_ah", "Charged Ah"),
        ("energy_in_wh", "Energy in (Wh)"),
        ("temperature_end", "End temperature (C)"),
        ("dv_dt_median_vps", "Median dV/dt (V/s)"),
    ]

    for ax, (col, label) in zip(axes.ravel(), plots):
        ax.scatter(cycle_df[col], y, s=8, alpha=0.45)
        ax.set_xlabel(label)
        ax.set_ylabel("Capacity label (Ah)")
        ax.grid(alpha=0.25)

    fig.savefig(OUTDIR / "05_capacity_vs_charge_features.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(cycle_df: pd.DataFrame) -> None:
    cols = [
        "capacity_ah",
        "sample_count",
        "duration_s",
        "charged_ah",
        "energy_in_wh",
        "power_mean_w",
        "voltage_start",
        "voltage_end",
        "temperature_start",
        "temperature_end",
        "dv_dt_median_vps",
        "dq_dv_median_ahpv",
    ]
    corr = cycle_df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("RW9 charge-cycle feature correlations (Spearman)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUTDIR / "06_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def write_report(cycle_df: pd.DataFrame) -> None:
    start_date = cycle_df["date_dt"].min()
    end_date = cycle_df["date_dt"].max()
    duration_days = cycle_df["elapsed_days"].max()
    corr_cols = [
        "charged_ah",
        "energy_in_wh",
        "power_mean_w",
        "temperature_end",
        "dv_dt_median_vps",
        "dq_dv_median_ahpv",
        "voltage_end",
    ]
    corr = cycle_df[["capacity_ah", *corr_cols]].corr(method="spearman")["capacity_ah"].drop("capacity_ah")
    corr_text = "\n".join(f"- `{name}`: {value:.3f}" for name, value in corr.sort_values(key=np.abs, ascending=False).items())

    report = f"""# RW9 Full Charging Report

## Scope
- Source charging rows: `rw9_datacapa.pkl`
- Source temperature signal: `RW9.mat`
- Capacity target used here: pkl `Capacity` label
- Verified alignment: charge-cycle counts and per-cycle sample counts match between MAT charge steps and pkl export

## Key Counts
- Charge cycles: {len(cycle_df):,}
- Unique capacity labels: {cycle_df['capacity_ah'].nunique():,}
- Time span: {duration_days:.2f} days
- First charge cycle: {start_date}
- Last charge cycle: {end_date}

## Cycle-Level Feature Ranges
- Charged Ah per cycle: {cycle_df['charged_ah'].min():.4f} to {cycle_df['charged_ah'].max():.4f}
- Energy in per cycle: {cycle_df['energy_in_wh'].min():.4f} to {cycle_df['energy_in_wh'].max():.4f}
- End temperature: {cycle_df['temperature_end'].min():.2f} to {cycle_df['temperature_end'].max():.2f} C
- Median dV/dt: {cycle_df['dv_dt_median_vps'].min():.6f} to {cycle_df['dv_dt_median_vps'].max():.6f} V/s
- Median dQ/dV: {cycle_df['dq_dv_median_ahpv'].min():.6f} to {cycle_df['dq_dv_median_ahpv'].max():.6f} Ah/V

## Strongest Spearman Correlations With Capacity
{corr_text}
"""
    (OUTDIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    cycle_df, sampled_df = load_rw9_charge_report_data()
    save_outputs(cycle_df, sampled_df)
    plot_voltage_temperature(sampled_df)
    plot_current_power(sampled_df)
    plot_derivatives(cycle_df)
    plot_capacity_over_time(cycle_df)
    plot_capacity_vs_features(cycle_df)
    plot_correlation_heatmap(cycle_df)
    write_report(cycle_df)


if __name__ == "__main__":
    main()
