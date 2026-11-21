"""MTP-3 (V7) versions of the descriptive MTP-2 figures.
Reads outputs/results/word_features_full.csv, applies the V7 filters
(dataset_tier >= 2, 20-2000 ms; no SD trim) and writes to outputs/figures:
  lmm_v7_word_length_vs_duration.png
  lmm_v7_duration_by_pos.png        (needs a POS column)
  lmm_v7_surprisal_distributions.png
  lmm_v7_duration_by_gender.png     (needs a gender column)
  lmm_v7_log_duration_hist.png
and prints the numbers you need for the captions (also saved to
outputs/results/lmm_v7_descriptive_summary.txt).
Run:  python3 make_v7_descriptive_figs.py [BASE_DIR]   (needs pandas, numpy, matplotlib)
Columns are auto-detected; any figure whose column is missing is skipped and
the script lists the column names so it can be fixed.
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/c/Users/MayurGaidhane/Downloads/indic_pipeline_full_dataset/indic_language_pipeline"
CSV = os.path.join(BASE, "outputs", "results", "word_features_full.csv")
FIG = os.path.join(BASE, "outputs", "figures")
SUMTXT = os.path.join(BASE, "outputs", "results", "lmm_v7_descriptive_summary.txt")
os.makedirs(FIG, exist_ok=True)

d = pd.read_csv(CSV, low_memory=False)
cols = list(d.columns)
lines = []
def pr(s=""):
    print(s); lines.append(s)

def pick(cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

c_dur = pick(["word_duration_ms"]); c_tier = pick(["dataset_tier"])
c_len = pick(["word_length_chars", "word_length"])
c_fwd = pick(["forward_surprisal"]); c_bwd = pick(["backward_surprisal"])
c_pos = pick(["upos", "pos", "stanza_upos", "word_upos", "pos_tag"])
c_gen = pick(["gender", "speaker_gender", "gender_file", "speaker_gender_file"])
c_reg = pick(["scenario_corrected"]); c_scn = pick(["scenario"])

d["reg"] = d[c_scn]
if c_reg is not None:
    sc = d[c_reg].fillna("")
    d["reg"] = d[c_scn].where(sc == "", sc)
for c in (c_dur, c_tier, c_len, c_fwd, c_bwd):
    if c is not None:
        d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[(d[c_tier] >= 2) & d[c_dur].between(20, 2000)].copy()
pr(f"Words after V7 filters (tier >= 2, 20-2000 ms): {len(d):,}")
pr(f"Columns used: length={c_len}, fwd={c_fwd}, bwd={c_bwd}, pos={c_pos}, gender={c_gen}")

B, O, A = "#2a78d6", "#eb6834", "#1baf7a"
INK, SUB, GRID = "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB})
CAVEAT = "Durations: about 76% of words are aligned as spn (noise), so treat short durations with caution."

def style(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def save(fig, name, caveat=True):
    if caveat:
        fig.text(0.01, 0.005, CAVEAT, fontsize=8, color=SUB)
    out = os.path.join(FIG, name)
    fig.savefig(out, facecolor="white"); plt.close(fig); pr("saved " + out)

def bars(labels, means, ns, title, fname, xlabel, color=B, rot=0):
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
    x = np.arange(len(labels))
    ax.bar(x, means, color=color)
    for xi, m in zip(x, means):
        ax.text(xi, m + max(means) * .015, f"{m:.0f}", ha="center", fontsize=8, color=INK)
    # label and n sit together in the tick label (two lines), so nothing overlaps
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn={n:,}" for l, n in zip(labels, ns)], rotation=rot,
                       ha="right" if rot else "center", fontsize=7 if len(labels) > 10 else 8.5)
    ax.set_ylabel("Mean word duration (ms)"); ax.set_xlabel(xlabel, labelpad=10)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    style(ax); fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, fname)

# 1. word length vs duration
if c_len:
    g = d[d[c_len].between(1, 15)].groupby(c_len)[c_dur].agg(["mean", "median", "count"])
    pr("\nWord length (characters): mean / median duration, n")
    for k, r in g.iterrows():
        pr(f"  {int(k):>2}  mean {r['mean']:.0f} ms  median {r['median']:.0f} ms  n={int(r['count']):,}")
    bars([str(int(k)) for k in g.index], g["mean"].values, g["count"].values,
         "V7 mean word duration by word length (characters)", "lmm_v7_word_length_vs_duration.png",
         "Word length (characters)")
else:
    pr("SKIPPED word length: column missing")

# 2. duration by POS
if c_pos:
    top = d[c_pos].value_counts().head(10).index
    g = d[d[c_pos].isin(top)].groupby(c_pos)[c_dur].agg(["mean", "count"]).sort_values("mean", ascending=False)
    pr("\nMean duration by the ten most frequent POS tags")
    for k, r in g.iterrows():
        pr(f"  {k:<6} mean {r['mean']:.0f} ms  n={int(r['count']):,}")
    bars(list(g.index), g["mean"].values, g["count"].values,
         "V7 mean word duration by part of speech (10 most frequent)", "lmm_v7_duration_by_pos.png",
         "Universal POS tag", color=O, rot=30)
else:
    pr("SKIPPED POS: no POS column found")

# 3. surprisal distributions
if c_fwd and c_bwd:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=200, sharey=False)
    for ax, c, nm, col in ((axes[0], c_bwd, "Backward surprisal", B), (axes[1], c_fwd, "Forward surprisal", O)):
        v = d[c].dropna(); v = v[v >= 0]
        ax.hist(v.clip(upper=50), bins=np.arange(0, 51, 1), color=col)
        ax.set_xlabel("bits (capped at 50)"); ax.set_title(nm, loc="left", fontsize=10, color=INK, fontweight="bold")
        style(ax)
        pr(f"\n{nm}: mean {v.mean():.2f}, median {v.median():.2f} bits, n={len(v):,}; "
           f"share in 0-1 bits {100*(v<1).mean():.1f}% ({int((v<1).sum()):,} words); above 50 bits {100*(v>50).mean():.2f}%")
    axes[0].set_ylabel("Words")
    fig.suptitle("V7 surprisal distributions", x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "lmm_v7_surprisal_distributions.png", caveat=False)
else:
    pr("SKIPPED surprisal: columns missing")

# 4. duration by gender
if c_gen:
    gs = d[c_gen].astype(str).str.strip().str.lower()
    g = d.assign(_g=gs).groupby("_g")[c_dur].agg(["mean", "median", "count"])
    g = g[g["count"] > 100]
    pr("\nDuration by gender")
    for k, r in g.iterrows():
        pr(f"  {k:<8} mean {r['mean']:.0f} ms  median {r['median']:.0f} ms  n={int(r['count']):,}")
    pr(f"  words with no gender label: {int((gs.isin(['nan','','none'])).sum()):,}")
    bars(list(g.index), g["mean"].values, g["count"].values,
         "V7 mean word duration by speaker gender", "lmm_v7_duration_by_gender.png", "Gender", color=A)
else:
    pr("SKIPPED gender: no gender column found")

# 5. log-duration histogram
fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=200)
ld = np.log10(d[c_dur].values)
ax.hist(ld, bins=np.arange(np.log10(20), np.log10(2000) + .02, .02), color=B)
for ms in (20, 30, 50):
    ax.axvline(np.log10(ms), color=INK, lw=.8, ls=(0, (4, 3)))
    ax.text(np.log10(ms), ax.get_ylim()[1] * .97, f"{ms} ms", fontsize=8, color=INK, ha="left", va="top")
ticks = [20, 50, 100, 200, 500, 1000, 2000]
ax.set_xticks(np.log10(ticks)); ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("Word duration (ms, log scale)"); ax.set_ylabel("Words")
ax.set_title("V7 log-duration histogram (tier >= 2, 20-2000 ms)", loc="left", fontsize=11, color=INK, fontweight="bold")
style(ax); fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, "lmm_v7_log_duration_hist.png")
for ms in (20, 30, 40):
    n = int(((d[c_dur] >= ms) & (d[c_dur] < ms + 10)).sum())
    pr(f"Words with duration {ms}-{ms+10} ms: {n:,} ({100*n/len(d):.1f}%)")

with open(SUMTXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
pr("\nSaved " + SUMTXT)
print("\nAll columns:", cols)