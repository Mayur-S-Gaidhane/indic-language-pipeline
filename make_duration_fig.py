"""Word duration by register, with the 50 ms line.
Reads outputs/results/word_features_full.csv, applies the V7 filters
(tier >= 2, 20-2000 ms; the +/-2.5 SD trim is not applied) and writes
outputs/figures/lmm_v7_duration_by_register.png.
Run:  python3 make_duration_fig.py   (needs pandas, matplotlib, numpy)
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/c/Users/MayurGaidhane/Downloads/indic_pipeline_full_dataset/indic_language_pipeline"
CSV = os.path.join(BASE, "outputs", "results", "word_features_full.csv")
FIG = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

d = pd.read_csv(CSV, usecols=lambda c: c in {"scenario", "scenario_corrected", "word_duration_ms", "dataset_tier"},
                low_memory=False)
sc = d["scenario_corrected"].fillna("") if "scenario_corrected" in d else pd.Series("", index=d.index)
d["reg"] = d["scenario"].where(sc == "", sc)
d["word_duration_ms"] = pd.to_numeric(d["word_duration_ms"], errors="coerce")
d["dataset_tier"] = pd.to_numeric(d["dataset_tier"], errors="coerce")
d = d[(d.dataset_tier >= 2) & d.word_duration_ms.between(20, 2000)]

B, O, A = "#2a78d6", "#eb6834", "#1baf7a"
INK, SUB, GRID = "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": SUB,
                     "xtick.color": SUB, "ytick.color": SUB})

regs = [("Conversation", B), ("Extempore", O), ("Read", A)]
XMAX = 600
bins = np.arange(20, XMAX + 10, 10)

fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), dpi=200, sharex=True)
ymax = 0
data = {}
for name, _ in regs:
    x = d.loc[d.reg == name, "word_duration_ms"].values
    h, _e = np.histogram(x, bins=bins)
    data[name] = (x, 100 * h / max(len(x), 1))
    ymax = max(ymax, data[name][1].max())

for ax, (name, col) in zip(axes, regs):
    x, share = data[name]
    ax.bar(bins[:-1], share, width=9, align="edge", color=col)
    ax.axvline(50, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_ylabel("% of words per 10 ms")
    under = 100 * np.mean(x < 50); over = 100 * np.mean(x > XMAX)
    ax.text(0.99, 0.93,
            f"{name}: n = {len(x):,}, median {np.median(x):.0f} ms\n"
            f"{under:.1f}% of words under 50 ms; {over:.1f}% above {XMAX} ms (not shown)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
axes[0].text(52, ymax * 1.05, "50 ms", fontsize=8.5, color=INK, va="top")
axes[-1].set_xlabel("Word duration (ms)")
axes[-1].set_xlim(20, XMAX)
fig.suptitle("V7 word duration by register (tier >= 2, 20-2000 ms)", x=0.01, ha="left",
             fontsize=11, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(FIG, "lmm_v7_duration_by_register.png")
fig.savefig(out, facecolor="white"); plt.close(fig)
print("saved", out)