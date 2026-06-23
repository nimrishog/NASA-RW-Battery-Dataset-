from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"c:\Users\nimri\Downloads\BatteryData")
OUTDIR = ROOT / "battery_data_hierarchy"
OUTDIR.mkdir(exist_ok=True)
OUT = OUTDIR / "battery_data_hierarchy.png"


def box(ax, x, y, w, h, text, fc, ec="#334155", fontsize=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold", color="#0f172a")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.2,
            color="#64748b",
        )
    )


fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")

ax.text(7.5, 8.05, "Battery Data Hierarchy Used in the Transformer Pipeline", ha="center", fontsize=18, weight="bold")
ax.text(
    7.5,
    7.65,
    "One model sample = one charge or discharge random-walk event. Rows inside the event become patch tokens.",
    ha="center",
    fontsize=11,
    color="#475569",
)

box(ax, 0.6, 6.2, 2.0, 0.9, "Raw file\nRW9.mat", "#dbeafe")
box(ax, 3.2, 6.2, 2.1, 0.9, "data.step\nlist of steps", "#dbeafe")
box(ax, 5.9, 6.2, 2.3, 0.9, "One step/event\ncharge or discharge", "#dcfce7")
box(ax, 8.8, 6.2, 2.4, 0.9, "Rows/samples\n1 Hz measurements", "#fef3c7")
box(ax, 11.8, 6.2, 2.4, 0.9, "Patch tokens\n30 rows per token", "#f3e8ff")

for x1, x2 in [(2.6, 3.2), (5.3, 5.9), (8.2, 8.8), (11.2, 11.8)]:
    arrow(ax, x1, 6.65, x2, 6.65)

box(ax, 0.8, 4.55, 2.8, 0.9, "Battery\nRW9 / RW10 / RW11", "#e0f2fe")
box(ax, 4.0, 4.55, 2.8, 0.9, "Event index\ncharge event #k\nor discharge event #k", "#dcfce7")
box(ax, 7.2, 4.55, 2.8, 0.9, "Row inside event\nVoltage, Current,\nTemperature, relTime", "#fef3c7")
box(ax, 10.4, 4.55, 2.8, 0.9, "Transformer input\nup to 11 tokens/event", "#f3e8ff")

for x1, x2 in [(3.6, 4.0), (6.8, 7.2), (10.0, 10.4)]:
    arrow(ax, x1, 5.0, x2, 5.0)

box(ax, 1.0, 2.65, 3.0, 1.05, "Reference-discharge\nbenchmark step", "#fee2e2")
box(ax, 4.8, 2.65, 3.1, 1.05, "Capacity label\nAh measured during\nreference discharge", "#fee2e2")
box(ax, 8.7, 2.65, 3.1, 1.05, "SOH target\nCapacity / initial\ncapacity × 100", "#fee2e2")
for x1, x2 in [(4.0, 4.8), (7.9, 8.7)]:
    arrow(ax, x1, 3.18, x2, 3.18)

ax.text(7.5, 2.15, "Label assignment", ha="center", fontsize=13, weight="bold", color="#991b1b")
ax.text(
    7.5,
    1.78,
    "Each charge/discharge event is paired with the next reference-discharge SOH checkpoint. Many events share the same label.",
    ha="center",
    fontsize=11,
    color="#475569",
)

box(ax, 2.0, 0.65, 3.1, 0.9, "Training samples\nRW9 + RW10 events", "#ecfeff")
box(ax, 6.0, 0.65, 3.1, 0.9, "Testing samples\nRW11 events", "#ecfeff")
box(ax, 10.0, 0.65, 3.1, 0.9, "Weighted loss\ninverse checkpoint\nblock size", "#ecfeff")
arrow(ax, 5.1, 1.1, 6.0, 1.1)
arrow(ax, 9.1, 1.1, 10.0, 1.1)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(OUT)
