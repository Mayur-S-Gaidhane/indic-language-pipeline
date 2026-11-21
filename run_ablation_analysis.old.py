#!/usr/bin/env python3
"""
run_ablation_analysis.py — Run all 8 hint modes for Sarvam & Gemini,
collect results, and generate analysis figures + summary CSV.

Usage:
  # Run evaluations + generate figures:
  python run_ablation_analysis.py

  # Skip evaluation (use existing results), just regenerate figures:
  python run_ablation_analysis.py --figures_only

  # Run only specific models:
  python run_ablation_analysis.py --models sarvam
  python run_ablation_analysis.py --models gemini

  # Run only specific hint modes:
  python run_ablation_analysis.py --hint_modes 1 2 3

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os, sys, subprocess, argparse, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(ROOT, "mtp3_youtube_eval_v2_enhanced.py")
OUT_DIR = os.path.join(ROOT, "outputs", "youtube_eval")
ANALYSIS_DIR = os.path.join(ROOT, "outputs", "ablation_analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Hint mode definitions (from eval script) ─────────────────
# Script's hint_mode numbering:
#   1: Structure Only    2: Theme Only       3: Keywords Only
#   4: Structure+Theme   5: Theme+Keywords   6: Structure+Keywords
#   7: All Three         8: None (Baseline)
HINT_MODE_NAMES = {
    1: "Structure Only",
    2: "Theme Only",
    3: "Keywords Only",
    4: "Structure + Theme",
    5: "Theme + Keywords",
    6: "Structure + Keywords",
    7: "All Three (K+T+S)",
    8: "None (Baseline)",
}

# Presentation mode mapping: M1–M8 for cleaner ordering
# M1=None, M2=K, M3=K+T, M4=K+S, M5=T+S, M6=T, M7=K+T+S, M8=S
PRES_MODE_ORDER = [8, 3, 5, 6, 4, 2, 7, 1]  # hint_mode → presentation order
PRES_MODE_LABELS = {
    8: "M1: None",
    3: "M2: Keywords (K)",
    5: "M3: K+T",
    6: "M4: K+S",
    4: "M5: T+S",
    2: "M6: Theme (T)",
    7: "M7: All (K+T+S)",
    1: "M8: Structure (S)",
}
PRES_MODE_SHORT = {
    8: "M1", 3: "M2", 5: "M3", 6: "M4",
    4: "M5", 2: "M6", 7: "M7", 1: "M8",
}

MODELS = ["sarvam", "gemini"]
HINT_MODES = list(range(1, 9))
METRICS = ["bleu", "rouge1", "rougeL", "bertscore_f1", "bertscore_f1_scaled", "embedding_cosine_sim"]
METRIC_DISPLAY = {
    "bleu": "BLEU",
    "rouge1": "ROUGE-1",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "bertscore_f1_scaled": "Scaled BERTScore F1",
    "embedding_cosine_sim": "Embedding Cosine Sim",
}

# Baselines for Scaled BERTScore F1
BASELINES = {
    "Gadkari (Conversational)": 0.6499,
    "HDTB (Written Text)": 0.6640,
}


# ══════════════════════════════════════════════════════════════
#  PHASE 1: Run evaluations
# ══════════════════════════════════════════════════════════════
def run_evaluations(models, hint_modes):
    """Run mtp3_youtube_eval_v2_enhanced.py for each model × hint_mode."""
    total = len(models) * len(hint_modes)
    done = 0
    for model in models:
        for hm in hint_modes:
            done += 1
            model_dir = os.path.join(OUT_DIR, f"{model}_v5_hm{hm}")
            select_csv = os.path.join(model_dir, "selected_responses.csv")

            # Skip if already completed
            if os.path.exists(select_csv):
                print(f"  [{done}/{total}] {model} hint_mode={hm} — already done, skipping")
                continue

            print(f"  [{done}/{total}] Running {model} hint_mode={hm} ...")
            cmd = [
                sys.executable, EVAL_SCRIPT,
                "--model", model,
                "--prompt_version", "5",
                "--hint_mode", str(hm),
                "--n_candidates", "10",
                "--context_window", "5",
                "--split", "test",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode != 0:
                    print(f"    [ERROR] Exit code {result.returncode}")
                    print(f"    stderr: {result.stderr[-500:]}" if result.stderr else "")
                else:
                    print(f"    Done.")
            except subprocess.TimeoutExpired:
                print(f"    [ERROR] Timed out after 30 minutes")
            except Exception as e:
                print(f"    [ERROR] {e}")


# ══════════════════════════════════════════════════════════════
#  PHASE 2: Collect results
# ══════════════════════════════════════════════════════════════
def collect_results(models, hint_modes):
    """Read AVERAGE row from each selected_responses.csv and build summary."""
    rows = []
    missing = []
    for model in models:
        for hm in hint_modes:
            model_dir = os.path.join(OUT_DIR, f"{model}_v5_hm{hm}")
            select_csv = os.path.join(model_dir, "selected_responses.csv")
            if not os.path.exists(select_csv):
                missing.append(f"{model}_v5_hm{hm}")
                continue

            df = pd.read_csv(select_csv)
            avg_row = df[df["pair_id"] == "AVERAGE"]
            if avg_row.empty:
                # Compute averages manually
                avgs = {}
                for m in METRICS:
                    if m in df.columns:
                        avgs[m] = df[m].astype(float).mean()
                    else:
                        avgs[m] = None
            else:
                avgs = {}
                for m in METRICS:
                    if m in avg_row.columns:
                        val = avg_row[m].values[0]
                        avgs[m] = float(val) if pd.notna(val) else None
                    else:
                        avgs[m] = None

            rows.append({
                "model": model,
                "hint_mode": hm,
                "hint_name": HINT_MODE_NAMES[hm],
                "pres_mode": PRES_MODE_SHORT[hm],
                **avgs
            })

    if missing:
        print(f"\n  [WARNING] Missing results for: {', '.join(missing)}")

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(ANALYSIS_DIR, "ablation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary saved: {summary_csv}")
    return summary_df


# ══════════════════════════════════════════════════════════════
#  PHASE 3: Generate figures
# ══════════════════════════════════════════════════════════════
def generate_figures(summary_df):
    """Generate all analysis figures using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("  [ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    # ── Figure 1: Clustered bar chart — Scaled BERTScore F1 ──
    print("  Generating Figure 1: Scaled BERTScore F1 comparison bar chart ...")
    fig1_bar_chart(summary_df, plt)

    # ── Figure 2: All 6 metrics comparison ──
    print("  Generating Figure 2: All metrics comparison grid ...")
    fig2_all_metrics(summary_df, plt)

    # ── Figure 3: Tier analysis heatmap ──
    print("  Generating Figure 3: Tier analysis heatmap ...")
    fig3_tier_heatmap(summary_df, plt)

    # ── Figure 4: Radar/spider chart ──
    print("  Generating Figure 4: Radar chart ...")
    fig4_radar(summary_df, plt)

    # ── Figure 5: Delta chart (Gemini − Sarvam) ──
    print("  Generating Figure 5: Model delta chart ...")
    fig5_delta(summary_df, plt)

    print(f"\n  All figures saved to: {ANALYSIS_DIR}/")


def _get_pres_order_data(summary_df, model, metric="bertscore_f1_scaled"):
    """Get data in presentation mode order (M1–M8)."""
    model_df = summary_df[summary_df["model"] == model].copy()
    labels = []
    values = []
    for hm in PRES_MODE_ORDER:
        row = model_df[model_df["hint_mode"] == hm]
        if not row.empty:
            labels.append(PRES_MODE_SHORT[hm])
            val = row[metric].values[0]
            values.append(float(val) if pd.notna(val) else 0)
    return labels, values


def fig1_bar_chart(summary_df, plt):
    """Clustered bar chart: Scaled BERTScore F1 by mode, Sarvam vs Gemini."""
    fig, ax = plt.subplots(figsize=(12, 6))

    labels_s, vals_s = _get_pres_order_data(summary_df, "sarvam")
    labels_g, vals_g = _get_pres_order_data(summary_df, "gemini")

    x = np.arange(len(labels_s))
    width = 0.35

    bars1 = ax.bar(x - width/2, vals_s, width, label="Sarvam 10B", color="#333333", edgecolor="white")
    bars2 = ax.bar(x + width/2, vals_g, width, label="Gemini 3.5 Flash Lite", color="#999999", edgecolor="white")

    ax.set_xlabel("Mode", fontsize=12)
    ax.set_ylabel("Scaled BERTScore F1", fontsize=12)
    ax.set_title("V5 Ablation: Scaled BERTScore F1 by Mode", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_s)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#666666")

    # Add tier background bands
    ax.axhspan(0.24, ax.get_ylim()[1], alpha=0.05, color="green", label="_Tier A")
    ax.axhspan(0.09, 0.24, alpha=0.05, color="orange", label="_Tier B")
    ax.axhspan(0, 0.09, alpha=0.05, color="red", label="_Tier C")

    plt.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "fig1_scaled_bertscore_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_all_metrics(summary_df, plt):
    """Grid of 6 subplots — one per metric, both models."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        labels_s, vals_s = _get_pres_order_data(summary_df, "sarvam", metric)
        labels_g, vals_g = _get_pres_order_data(summary_df, "gemini", metric)

        x = np.arange(len(labels_s))
        width = 0.35

        ax.bar(x - width/2, vals_s, width, label="Sarvam 10B", color="#333333")
        ax.bar(x + width/2, vals_g, width, label="Gemini 3.5 Flash Lite", color="#999999")

        ax.set_title(METRIC_DISPLAY[metric], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_s, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("V5 Ablation: All Metrics Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "fig2_all_metrics_grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_tier_heatmap(summary_df, plt):
    """Heatmap of Scaled BERTScore F1 — modes × models with tier coloring."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ["sarvam", "gemini"]
    model_labels = ["Sarvam 10B", "Gemini 3.5 Flash Lite"]
    data = []
    mode_labels = []

    for hm in PRES_MODE_ORDER:
        mode_labels.append(PRES_MODE_SHORT[hm])
        row = []
        for model in models:
            val = summary_df[(summary_df["model"] == model) & (summary_df["hint_mode"] == hm)]["bertscore_f1_scaled"]
            row.append(float(val.values[0]) if not val.empty and pd.notna(val.values[0]) else 0)
        data.append(row)

    data = np.array(data)
    im = ax.imshow(data.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.4)

    ax.set_xticks(range(len(mode_labels)))
    ax.set_xticklabels(mode_labels)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)

    # Annotate cells
    for i in range(len(mode_labels)):
        for j in range(len(model_labels)):
            val = data[i, j]
            color = "white" if val > 0.2 else "black"
            ax.text(i, j, f"{val:.3f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_title("Scaled BERTScore F1 Heatmap — Modes × Models", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Scaled BERTScore F1")
    plt.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "fig3_tier_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_radar(summary_df, plt):
    """Radar chart comparing all 6 metrics for best mode of each model."""
    from math import pi

    # Find best mode per model (by scaled BERTScore F1)
    best = {}
    for model in MODELS:
        mdf = summary_df[summary_df["model"] == model]
        best_row = mdf.loc[mdf["bertscore_f1_scaled"].idxmax()]
        best[model] = best_row

    categories = list(METRIC_DISPLAY.values())
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, color, label in [("sarvam", "#333333", "Sarvam 10B"), ("gemini", "#999999", "Gemini 3.5 Flash Lite")]:
        row = best[model]
        values = [float(row[m]) if pd.notna(row[m]) else 0 for m in METRICS]
        values += values[:1]
        hm = int(row["hint_mode"])
        ax.plot(angles, values, "o-", linewidth=2, label=f"{label} (Best: {PRES_MODE_SHORT[hm]})", color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Best Mode Comparison — All Metrics", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "fig4_radar_best_mode.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_delta(summary_df, plt):
    """Delta chart: Gemini − Sarvam for Scaled BERTScore F1 per mode."""
    fig, ax = plt.subplots(figsize=(10, 5))

    mode_labels = []
    deltas = []
    for hm in PRES_MODE_ORDER:
        s_val = summary_df[(summary_df["model"] == "sarvam") & (summary_df["hint_mode"] == hm)]["bertscore_f1_scaled"]
        g_val = summary_df[(summary_df["model"] == "gemini") & (summary_df["hint_mode"] == hm)]["bertscore_f1_scaled"]
        if not s_val.empty and not g_val.empty:
            mode_labels.append(PRES_MODE_SHORT[hm])
            deltas.append(float(g_val.values[0]) - float(s_val.values[0]))

    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    bars = ax.bar(range(len(deltas)), deltas, color=colors, edgecolor="white")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(mode_labels)))
    ax.set_xticklabels(mode_labels)
    ax.set_ylabel("Gemini − Sarvam (Scaled BERTScore F1)")
    ax.set_title("Model Advantage by Mode (Positive = Gemini Better)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, (bar, d) in enumerate(zip(bars, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2, d + (0.003 if d >= 0 else -0.01),
                f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, "fig5_model_delta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  PHASE 4: Generate text summary
# ══════════════════════════════════════════════════════════════
def generate_summary_report(summary_df):
    """Generate a text report with tier analysis and key findings."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  V5 ABLATION ANALYSIS — COMPLETE REPORT")
    report_lines.append("=" * 70)

    # Per-model tables
    for model in MODELS:
        mdf = summary_df[summary_df["model"] == model].copy()
        model_name = "Sarvam 10B" if model == "sarvam" else "Gemini 3.5 Flash Lite Flash"
        report_lines.append(f"\n{'─' * 70}")
        report_lines.append(f"  {model_name}")
        report_lines.append(f"{'─' * 70}")
        report_lines.append(f"\n  {'Mode':<10} {'Hint':<25} {'BLEU':>8} {'R-1':>8} {'R-L':>8} {'BSF1':>8} {'SBF1':>8} {'ECS':>8}")
        report_lines.append(f"  {'─'*10} {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        for hm in PRES_MODE_ORDER:
            row = mdf[mdf["hint_mode"] == hm]
            if row.empty:
                continue
            r = row.iloc[0]
            report_lines.append(
                f"  {PRES_MODE_SHORT[hm]:<10} {HINT_MODE_NAMES[hm]:<25} "
                f"{r.get('bleu', 0):>8.4f} {r.get('rouge1', 0):>8.4f} "
                f"{r.get('rougeL', 0):>8.4f} {r.get('bertscore_f1', 0):>8.4f} "
                f"{r.get('bertscore_f1_scaled', 0):>8.4f} {r.get('embedding_cosine_sim', 0):>8.4f}"
            )

    # Tier analysis
    report_lines.append(f"\n{'=' * 70}")
    report_lines.append("  TIER ANALYSIS (Scaled BERTScore F1)")
    report_lines.append(f"{'=' * 70}")

    tiers = {"A (> 0.24)": [], "B (0.09–0.19)": [], "C (< 0.09)": []}
    for hm in PRES_MODE_ORDER:
        # Use Gemini as reference for tier assignment
        g_val = summary_df[(summary_df["model"] == "gemini") & (summary_df["hint_mode"] == hm)]["bertscore_f1_scaled"]
        s_val = summary_df[(summary_df["model"] == "sarvam") & (summary_df["hint_mode"] == hm)]["bertscore_f1_scaled"]
        avg = 0
        if not g_val.empty and not s_val.empty:
            avg = (float(g_val.values[0]) + float(s_val.values[0])) / 2

        if avg > 0.24:
            tiers["A (> 0.24)"].append(PRES_MODE_SHORT[hm])
        elif avg >= 0.09:
            tiers["B (0.09–0.19)"].append(PRES_MODE_SHORT[hm])
        else:
            tiers["C (< 0.09)"].append(PRES_MODE_SHORT[hm])

    for tier, modes in tiers.items():
        report_lines.append(f"\n  Tier {tier}: {', '.join(modes) if modes else '(none)'}")

    # Key findings
    report_lines.append(f"\n{'=' * 70}")
    report_lines.append("  KEY FINDINGS")
    report_lines.append(f"{'=' * 70}")

    for model in MODELS:
        mdf = summary_df[summary_df["model"] == model]
        if not mdf.empty:
            best = mdf.loc[mdf["bertscore_f1_scaled"].idxmax()]
            worst = mdf.loc[mdf["bertscore_f1_scaled"].idxmin()]
            model_name = "Sarvam 10B" if model == "sarvam" else "Gemini 3.5 Flash Lite"
            report_lines.append(f"\n  {model_name}:")
            report_lines.append(f"    Best:  {PRES_MODE_SHORT[int(best['hint_mode'])]} ({HINT_MODE_NAMES[int(best['hint_mode'])]}) = {best['bertscore_f1_scaled']:.4f}")
            report_lines.append(f"    Worst: {PRES_MODE_SHORT[int(worst['hint_mode'])]} ({HINT_MODE_NAMES[int(worst['hint_mode'])]}) = {worst['bertscore_f1_scaled']:.4f}")

    report_lines.append(f"\n{'=' * 70}")

    report_path = os.path.join(ANALYSIS_DIR, "ablation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report saved: {report_path}")

    # Also print to console
    for line in report_lines:
        print(line)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="V5 Ablation: run all modes, collect results, generate figures")
    parser.add_argument("--figures_only", action="store_true",
                        help="Skip evaluation, just regenerate figures from existing results")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        choices=MODELS, help="Models to evaluate")
    parser.add_argument("--hint_modes", nargs="+", type=int, default=HINT_MODES,
                        help="Hint modes to run (1-8)")
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--context_window", type=int, default=5)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print("=" * 65)
    print("  V5 ABLATION ANALYSIS PIPELINE")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Hint modes: {args.hint_modes}")
    print(f"  Figures only: {args.figures_only}")
    print("=" * 65)

    # Phase 1: Run evaluations
    if not args.figures_only:
        print("\n── Phase 1: Running evaluations ──")
        run_evaluations(args.models, args.hint_modes)
    else:
        print("\n── Phase 1: Skipped (--figures_only) ──")

    # Phase 2: Collect results
    print("\n── Phase 2: Collecting results ──")
    summary_df = collect_results(args.models, args.hint_modes)

    if summary_df.empty:
        print("\n  [ERROR] No results found. Run evaluations first (without --figures_only).")
        sys.exit(1)

    # Phase 3: Generate figures
    print("\n── Phase 3: Generating figures ──")
    generate_figures(summary_df)

    # Phase 4: Generate report
    print("\n── Phase 4: Generating summary report ──")
    generate_summary_report(summary_df)

    print("\n" + "=" * 65)
    print("  DONE! Outputs in:")
    print(f"    {ANALYSIS_DIR}/")
    print(f"    - ablation_summary.csv       (all metrics, all modes)")
    print(f"    - ablation_report.txt        (text summary + tiers)")
    print(f"    - fig1_scaled_bertscore_comparison.png")
    print(f"    - fig2_all_metrics_grid.png")
    print(f"    - fig3_tier_heatmap.png")
    print(f"    - fig4_radar_best_mode.png")
    print(f"    - fig5_model_delta.png")
    print("=" * 65)


if __name__ == "__main__":
    main()