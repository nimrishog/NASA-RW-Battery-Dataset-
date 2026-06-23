from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Inches, Pt


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
SUITE_DIR = ROOT / "charge_discharge_full_cycle_suite"
XAI_DIR = SUITE_DIR / "interpretability_shap_lime"
OUTDIR = ROOT / "ppt_charge_discharge_xai_summary"
ASSET_DIR = OUTDIR / "assets"
PPT_PATH = OUTDIR / "charge_discharge_feature_token_transformer_xai_summary.pptx"

COLORS = {
    "navy": RGBColor(23, 43, 77),
    "muted": RGBColor(95, 102, 116),
    "blue": RGBColor(42, 108, 176),
    "orange": RGBColor(226, 116, 49),
    "green": RGBColor(47, 133, 90),
    "purple": RGBColor(107, 78, 161),
    "red": RGBColor(190, 75, 73),
    "light": RGBColor(245, 247, 250),
    "line": RGBColor(210, 216, 225),
    "white": RGBColor(255, 255, 255),
}

BATTERY_COLORS = {"RW9": "#1f77b4", "RW10": "#ff7f0e", "RW11": "#2ca02c", "RW12": "#d62728"}


def ensure_dirs() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d-%b-%Y %H:%M:%S", errors="coerce")


def load_reference_capacity(battery: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "Analysis" / battery / "reference_capacity_summary.csv")
    df["date"] = parse_date(df["date"])
    out = (
        df.groupby("charge_cycle_count_before_reference", as_index=False)
        .agg(date=("date", "min"), capacity_ah=("capacity_ah", "mean"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    init = out.loc[out["charge_cycle_count_before_reference"].eq(0), "capacity_ah"].iloc[0]
    out["soh_percent"] = out["capacity_ah"] / init * 100.0
    out["battery"] = battery
    return out


def create_capacity_soh_plot() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for battery in ("RW9", "RW10", "RW11", "RW12"):
        df = load_reference_capacity(battery)
        axes[0].plot(df["date"], df["capacity_ah"], marker="o", ms=3.2, lw=1.7, label=battery, color=BATTERY_COLORS[battery])
        axes[1].plot(df["date"], df["soh_percent"], marker="o", ms=3.2, lw=1.7, label=battery, color=BATTERY_COLORS[battery])
    axes[0].set_title("Reference-discharge capacity labels")
    axes[0].set_ylabel("Capacity (Ah)")
    axes[1].set_title("SOH target labels")
    axes[1].set_ylabel("SOH (%)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    axes[0].legend(ncol=4, frameon=False, loc="upper right")
    fig.suptitle("Benchmark labels are sparse reference-discharge checkpoints", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSET_DIR / "capacity_soh_reference_labels.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def create_event_coverage_plot() -> Path:
    event_summary = pd.read_csv(SUITE_DIR / "event_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    pivot = event_summary.pivot(index="battery", columns="event_type", values="events").loc[["RW9", "RW10", "RW11"]]
    x = np.arange(len(pivot))
    width = 0.36
    axes[0].bar(x - width / 2, pivot["charge"], width, label="charge events", color="#2a9d8f")
    axes[0].bar(x + width / 2, pivot["discharge"], width, label="discharge events", color="#e76f51")
    axes[0].set_xticks(x, pivot.index)
    axes[0].set_ylabel("Events")
    axes[0].set_title("Charge/discharge event count")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    rows = []
    for battery in ("RW9", "RW10", "RW11"):
        ref = load_reference_capacity(battery)
        rows.append({"battery": battery, "unique_reference_checkpoints": int((ref["charge_cycle_count_before_reference"] > 0).sum())})
    chk = pd.DataFrame(rows)
    axes[1].bar(chk["battery"], chk["unique_reference_checkpoints"], color="#457b9d")
    axes[1].set_ylabel("Reference SOH checkpoints")
    axes[1].set_title("Sparse labels behind many events")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Adding discharge increases input events, not independent benchmark labels", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = ASSET_DIR / "charge_discharge_event_coverage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def create_top_shap_table_plot() -> Path:
    tf = pd.read_csv(XAI_DIR / "feature_token_transformer" / "global_shap_feature_importance.csv").head(10)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    top = tf.iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y, top["mean_abs_shap_combined"], color="#6b4ea1")
    ax.set_yticks(y, top["feature"])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top global SHAP drivers: Feature-token transformer")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out = ASSET_DIR / "top_transformer_shap_features.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def add_title(slide, title: str, subtitle: str | None = None) -> None:
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.25), Inches(12.2), Inches(0.55))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = COLORS["navy"]
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.58), Inches(0.80), Inches(12.0), Inches(0.38))
        sp = sub.text_frame.paragraphs[0]
        sr = sp.add_run()
        sr.text = subtitle
        sr.font.size = Pt(11)
        sr.font.color.rgb = COLORS["muted"]


def add_footer(slide, text: str = "NASA RW battery SOH modeling | charge + discharge feature-token transformer") -> None:
    box = slide.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(12.2), Inches(0.22))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = text
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(135, 142, 154)


def add_textbox(slide, x: float, y: float, w: float, h: float, text: str, size: int = 13, bold: bool = False, color=COLORS["navy"]) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_bullets(slide, x: float, y: float, w: float, h: float, bullets: list[str], size: int = 12) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(size)
        p.font.color.rgb = COLORS["navy"]


def add_image(slide, path: Path, x: float, y: float, w: float, h: float | None = None) -> None:
    if h is None:
        with Image.open(path) as im:
            aspect = im.height / im.width
        h = w * aspect
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def add_block(slide, x: float, y: float, w: float, h: float, text: str, fill: RGBColor, font_size: int = 11) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = COLORS["line"]
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = True
    run.font.color.rgb = COLORS["white"]


def add_arrow(slide, x1: float, y1: float, x2: float, y2: float) -> None:
    line = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = RGBColor(120, 128, 140)
    line.line.width = Pt(1.6)
    line.line.end_arrowhead = True


def add_table(slide, df: pd.DataFrame, x: float, y: float, w: float, h: float, font_size: int = 9) -> None:
    rows, cols = df.shape[0] + 1, df.shape[1]
    table = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for col_idx, col in enumerate(df.columns):
        cell = table.cell(0, col_idx)
        cell.text = str(col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["navy"]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(font_size)
            p.font.bold = True
            p.font.color.rgb = COLORS["white"]
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(250, 251, 253) if row_idx % 2 == 0 else RGBColor(238, 242, 247)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size)
                p.font.color.rgb = COLORS["navy"]


def build_presentation() -> None:
    ensure_dirs()
    capacity_plot = create_capacity_soh_plot()
    coverage_plot = create_event_coverage_plot()
    shap_top_plot = create_top_shap_table_plot()

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    summary = pd.read_csv(SUITE_DIR / "combined_charge_vs_charge_discharge_summary.csv")
    cd_summary = pd.read_csv(SUITE_DIR / "model_suite_summary.csv")
    best_metrics = json.loads((SUITE_DIR / "feature_tokens_h4_d64_l2" / "metrics.json").read_text())
    local_instances = pd.read_csv(XAI_DIR / "local_instances_used.csv")
    top_shap = pd.read_csv(XAI_DIR / "feature_token_transformer" / "global_shap_feature_importance.csv").head(8)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Charge + Discharge SOH Prediction With XAI", "Feature-token pure transformer trained on RW9+RW10 and tested on RW11")
    add_textbox(slide, 0.75, 1.35, 5.6, 0.8, "Goal", 18, True, COLORS["green"])
    add_bullets(
        slide,
        0.75,
        1.9,
        5.8,
        2.0,
        [
            "Predict SOH (%) from charge and discharge random-walk events.",
            "Use reference-discharge capacity checkpoints as benchmark labels.",
            "Explain the best transformer using global SHAP and local LIME.",
        ],
        15,
    )
    add_textbox(slide, 7.0, 1.35, 5.4, 0.8, "Best Charge+Discharge Transformer", 18, True, COLORS["purple"])
    add_bullets(
        slide,
        7.0,
        1.9,
        5.6,
        2.6,
        [
            "Feature-token pure transformer.",
            "63 engineered event features grouped into 8 feature tokens.",
            "4 heads, d_model=64, 2 layers, dropout=0.12.",
            "RW11 event-level R² = 0.9331, MAE = 3.20, RMSE = 3.71.",
        ],
        15,
    )
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Benchmark Capacity and SOH Labels", "The target is still the sparse reference-discharge benchmark capacity, even when inputs include charge + discharge events.")
    add_image(slide, capacity_plot, 0.65, 1.18, 12.0, 4.95)
    add_textbox(slide, 0.85, 6.32, 11.5, 0.42, "Key point: adding discharge increases input events, but the SOH labels still come from sparse reference-discharge checkpoints.", 12, True, COLORS["red"])
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Event Coverage After Adding Discharge", "Train = RW9 + RW10, test = RW11.")
    add_image(slide, coverage_plot, 0.75, 1.25, 11.8, 4.05)
    coverage_df = pd.DataFrame(
        [
            ["Train", "RW9 + RW10", "94,965 events"],
            ["Test", "RW11", "46,490 events"],
            ["Labels", "RW11 reference checkpoints", "38 benchmark labels"],
        ],
        columns=["Split", "Battery", "Count"],
    )
    add_table(slide, coverage_df, 2.0, 5.55, 9.2, 1.1, 10)
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Feature-Token Transformer Pipeline", "Full-event engineered features are the transformer tokens; there is no separate MLP branch.")
    y = 1.45
    add_block(slide, 0.45, y, 1.65, 0.8, "Raw MAT\nsteps", COLORS["muted"])
    add_arrow(slide, 2.1, y + 0.4, 2.55, y + 0.4)
    add_block(slide, 2.55, y, 1.85, 0.8, "Extract\ncharge + discharge", COLORS["blue"])
    add_arrow(slide, 4.4, y + 0.4, 4.85, y + 0.4)
    add_block(slide, 4.85, y, 1.85, 0.8, "Assign next\nreference SOH", COLORS["green"])
    add_arrow(slide, 6.7, y + 0.4, 7.15, y + 0.4)
    add_block(slide, 7.15, y, 1.85, 0.8, "Engineer\n63 features", COLORS["orange"])
    add_arrow(slide, 9.0, y + 0.4, 9.45, y + 0.4)
    add_block(slide, 9.45, y, 1.65, 0.8, "8 feature\ntokens", COLORS["purple"])
    add_arrow(slide, 11.1, y + 0.4, 11.55, y + 0.4)
    add_block(slide, 11.55, y, 1.35, 0.8, "SOH\nprediction", COLORS["red"])
    add_bullets(
        slide,
        0.75,
        3.05,
        5.8,
        2.7,
        [
            "Input sample = one charge or discharge random-walk event.",
            "Event-level features include throughput, voltage response, current statistics, temperature response, voltage-window timing, dQ/dV, dV/dQ, and distributional features.",
            "Each group of 8 standardized features becomes one transformer token.",
        ],
        12,
    )
    params = pd.DataFrame(
        [
            ["d_model", best_metrics["spec"]["d_model"]],
            ["heads", best_metrics["spec"]["nhead"]],
            ["layers", best_metrics["spec"]["num_layers"]],
            ["dropout", best_metrics["spec"]["dropout"]],
            ["learning rate", best_metrics["spec"]["learning_rate"]],
            ["weighted loss", "inverse benchmark-block MSE"],
        ],
        columns=["Parameter", "Value"],
    )
    add_table(slide, params, 7.1, 3.05, 5.2, 2.6, 10)
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Model Results: Charge Only vs Charge + Discharge", "The best charge+discharge transformer is the feature-token pure transformer.")
    add_image(slide, SUITE_DIR / "combined_charge_vs_charge_discharge_summary.png", 0.65, 1.15, 12.1, 4.6)
    best_table = pd.DataFrame(
        [
            ["Feature-token transformer", "charge + discharge", "0.9331", "3.20", "3.71"],
            ["LightGBM", "charge + discharge", "0.9284", "3.29", "3.84"],
            ["Patch-only transformer", "charge + discharge", "0.9211", "3.35", "4.03"],
            ["LightGBM", "charge only", "0.9576", "2.55", "2.97"],
        ],
        columns=["Model", "Input", "R²", "MAE", "RMSE"],
    )
    add_table(slide, best_table, 1.25, 5.9, 10.8, 1.1, 9)
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Best Transformer: Training and RW11 Prediction", "Feature-token pure transformer on charge+discharge events.")
    add_image(slide, SUITE_DIR / "feature_tokens_h4_d64_l2" / "training_dashboard.png", 0.55, 1.18, 6.25, 4.7)
    add_image(slide, SUITE_DIR / "feature_tokens_h4_d64_l2" / "prediction_report.png", 6.95, 1.18, 5.75, 4.7)
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Global SHAP: Which Features Drive the Transformer?", "SHAP computed for charge and discharge RW11 events separately and combined.")
    add_image(slide, XAI_DIR / "feature_token_transformer" / "global_shap_bar_by_event_type.png", 0.65, 1.15, 6.25, 5.2)
    add_image(slide, shap_top_plot, 7.0, 1.25, 5.7, 4.9)
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Global SHAP Beeswarm: Charge vs Discharge", "Separate beeswarm plots show whether drivers behave differently by event type.")
    add_image(slide, XAI_DIR / "feature_token_transformer" / "global_shap_beeswarm_charge.png", 0.55, 1.1, 6.2, 5.45)
    add_image(slide, XAI_DIR / "feature_token_transformer" / "global_shap_beeswarm_discharge.png", 6.85, 1.1, 6.2, 5.45)
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Local LIME: One Charge Event and One Discharge Event", "LIME explains one representative RW11 event of each type.")
    add_image(slide, XAI_DIR / "feature_token_transformer" / "local_lime_charge.png", 0.65, 1.15, 6.0, 4.95)
    add_image(slide, XAI_DIR / "feature_token_transformer" / "local_lime_discharge.png", 6.85, 1.15, 6.0, 4.95)
    local_tbl = local_instances[["event_type", "event_order", "assigned_checkpoint_cycle", "actual_soh_percent", "transformer_prediction"]].copy()
    local_tbl["actual_soh_percent"] = local_tbl["actual_soh_percent"].map(lambda x: f"{x:.2f}")
    local_tbl["transformer_prediction"] = local_tbl["transformer_prediction"].map(lambda x: f"{x:.2f}")
    add_table(slide, local_tbl, 2.2, 6.2, 8.9, 0.75, 8)
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "XAI Interpretation Summary", "The dominant drivers include both chronology and event-response features.")
    shap_tbl = top_shap[["feature", "mean_abs_shap_combined", "mean_abs_shap_charge", "mean_abs_shap_discharge"]].copy()
    for col in ["mean_abs_shap_combined", "mean_abs_shap_charge", "mean_abs_shap_discharge"]:
        shap_tbl[col] = shap_tbl[col].map(lambda x: f"{x:.3f}")
    shap_tbl.columns = ["Feature", "Combined", "Charge", "Discharge"]
    add_table(slide, shap_tbl, 0.65, 1.2, 7.2, 4.4, 8)
    add_bullets(
        slide,
        8.25,
        1.35,
        4.4,
        3.3,
        [
            "time_start, time_end, and label_charge_count encode battery aging chronology.",
            "sample_count, duration, charge_throughput, and energy_throughput describe how much operation occurred in the event.",
            "temperature_start and related thermal features capture operating-condition differences between events.",
            "Because labels are sparse checkpoints, XAI should be interpreted at benchmark-block level, not as 46,490 independent SOH tests.",
        ],
        12,
    )
    add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Final Takeaways", "What to report from the latest experiment.")
    add_bullets(
        slide,
        0.85,
        1.25,
        11.8,
        4.7,
        [
            "Best charge+discharge transformer: feature-token pure transformer, RW11 R² = 0.9331.",
            "The model uses 63 engineered event-level features grouped into 8 transformer tokens.",
            "Adding discharge roughly doubles event count, but the independent SOH labels remain sparse reference-discharge checkpoints.",
            "Patch-augmented all-feature transformer did not improve over the feature-token transformer; the compact feature-token design worked better.",
            "SHAP and LIME show that chronology-related features are strong, so claims should state that time/history is available to the model.",
        ],
        16,
    )
    add_footer(slide)

    prs.save(PPT_PATH)
    print(PPT_PATH)


if __name__ == "__main__":
    build_presentation()
