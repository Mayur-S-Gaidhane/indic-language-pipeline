#!/usr/bin/env python3
"""
mtp3_step1_framework.py — MTP-3 Objective 4 + Step 6 (partial)

PURPOSE
-------
1. Compute Cognitive Constraint Score (CCS) reference distributions
   from Pipeline V2 full dataset (word_features.csv)
2. Extract context-response conversation pairs from segments.csv
3. Output: ccs_reference.csv, conversation_pairs.csv, summary stats

THEORETICAL GROUNDING
---------------------
CCS weights come from LMM V7 M4 (Full Linguistic model):
  IC            β = +0.0018 ns   (Integration Cost — DLT)
  SC            β = +0.0033 *    (Storage Cost — DLT)
  fwd_surprisal β = +0.0006 ns   (Forward Surprisal)
  bwd_surprisal β = -0.0011 ns   (Backward Surprisal)

dep_distance removed per Prof Ranjan's Sept 29 feedback (= depth_test).
IC/SC corrected: both-direction, later-token assignment, non-PUNCT
interveners (verified 100% match against Prof's dlt.py).

These coefficients represent the effect of each cognitive predictor
on log(word_duration_ms) in Hindi conversational speech, controlling
for word_length, log_freq, dysfluency, and random intercepts for
speaker, word, and modality (n=406,164 words, 289 speakers).

The CCS for a response = weighted sum of mean z-scored features
across all words in that response. A response whose CCS falls
within the human reference distribution is "cognitively natural."

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur

INPUT
-----
  word_features.csv  (614,351 × 128)  — from Pipeline V2 full dataset
  segments.csv       (32,823  × 129)  — from Pipeline V2 full dataset

OUTPUT
------
  outputs/mtp3/ccs_reference.csv         — CCS per segment (reference)
  outputs/mtp3/ccs_reference_stats.txt   — Distribution statistics
  outputs/mtp3/conversation_pairs.csv    — Context-response pairs
  outputs/mtp3/pair_stats.txt            — Pair extraction summary
  outputs/figures/ccs_distribution.png   — CCS histogram per scenario
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(ROOT, "outputs", "results")
MTP3_DIR     = os.path.join(ROOT, "outputs", "mtp3")
FIGURES_DIR  = os.path.join(ROOT, "outputs", "figures")

# Input files — Pipeline V2 full dataset
WORD_CSV     = os.path.join(RESULTS_DIR, "word_features_full.csv")
SEG_CSV      = os.path.join(RESULTS_DIR, "segments_full.csv")

# If the above don't exist, try alternate names
WORD_CSV_ALT = os.path.join(RESULTS_DIR, "word_features_v4.csv")
SEG_CSV_ALT  = os.path.join(RESULTS_DIR,
                  "indicvoices_v1_hindi_segments_v4.csv")

# Output files
CCS_REF_CSV  = os.path.join(MTP3_DIR, "ccs_reference.csv")
CCS_STATS    = os.path.join(MTP3_DIR, "ccs_reference_stats.txt")
PAIRS_CSV    = os.path.join(MTP3_DIR, "conversation_pairs.csv")
PAIRS_STATS  = os.path.join(MTP3_DIR, "pair_stats.txt")
CCS_FIG      = os.path.join(FIGURES_DIR, "ccs_distribution.png")

for d in [MTP3_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── CCS Weights from LMM V7 M4 (Full Linguistic) ────────────
# Source: lmm_results_v7.txt, Model M4: Full Linguistic
# dep_distance removed per Prof Ranjan's Sept 29 feedback
CCS_WEIGHTS = {
    "IC":                  +0.0018,   # Integration Cost (ns, p=0.18)
    "SC":                  +0.0033,   # Storage Cost (* p=0.02)
    "forward_surprisal":   +0.0006,   # Forward Surprisal (ns)
    "backward_surprisal":  -0.0011,   # Backward Surprisal (ns)
}

# Lexical weights (from LMM V7 M4)
LEXICAL_WEIGHTS = {
    "word_length_chars":   +0.0339,   # *** p<0.001
    "log_word_freq":       -0.0489,   # *** p<0.001
}

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ══════════════════════════════════════════════════════════════
#  PART 1: CCS REFERENCE DISTRIBUTION
# ══════════════════════════════════════════════════════════════

def compute_ccs_reference(word_df):
    """
    Compute CCS for each segment in the corpus.
    Returns DataFrame with one row per segment + CCS values.
    """
    pr()
    pr("=" * 65)
    pr("  PART 1: CCS REFERENCE DISTRIBUTION")
    pr("  Computing Cognitive Constraint Score per segment")
    pr("=" * 65)

    # ── Numeric conversion ────────────────────────────────────
    num_cols = list(CCS_WEIGHTS.keys()) + list(LEXICAL_WEIGHTS.keys())
    for col in num_cols:
        if col in word_df.columns:
            word_df[col] = pd.to_numeric(
                word_df[col], errors="coerce")

    # ── Quality filter ────────────────────────────────────────
    if "dataset_tier" in word_df.columns:
        word_df["dataset_tier"] = pd.to_numeric(
            word_df["dataset_tier"], errors="coerce")
        word_df = word_df[word_df["dataset_tier"] >= 2].copy()
        pr(f"  After tier >= 2 filter: {len(word_df):,} words")

    if "word_duration_ms" in word_df.columns:
        word_df["word_duration_ms"] = pd.to_numeric(
            word_df["word_duration_ms"], errors="coerce")
        word_df = word_df[
            (word_df["word_duration_ms"] >= 20) &
            (word_df["word_duration_ms"] <= 2000)
        ].copy()
        pr(f"  After duration 20-2000ms: {len(word_df):,} words")

    # ── Z-score all CCS features (corpus-level) ──────────────
    pr()
    pr("  Z-scoring CCS features (corpus-level):")
    z_cols = {}
    for feat in list(CCS_WEIGHTS.keys()) + list(LEXICAL_WEIGHTS.keys()):
        if feat in word_df.columns:
            mu = word_df[feat].mean()
            sd = word_df[feat].std()
            z_name = f"{feat}_z"
            if sd > 0:
                word_df[z_name] = (word_df[feat] - mu) / sd
            else:
                word_df[z_name] = 0.0
            z_cols[feat] = z_name
            pr(f"    {feat:<25} mean={mu:.3f}  sd={sd:.3f}")

    # ── Compute word-level CCS ────────────────────────────────
    pr()
    pr("  Computing word-level CCS:")
    pr(f"    Weights: {CCS_WEIGHTS}")

    word_df["ccs_word"] = 0.0
    for feat, weight in CCS_WEIGHTS.items():
        z_name = z_cols.get(feat)
        if z_name and z_name in word_df.columns:
            word_df["ccs_word"] += weight * word_df[z_name].fillna(0)

    # Also compute lexical score
    word_df["lexical_word"] = 0.0
    for feat, weight in LEXICAL_WEIGHTS.items():
        z_name = z_cols.get(feat)
        if z_name and z_name in word_df.columns:
            word_df["lexical_word"] += weight * word_df[z_name].fillna(0)

    # Combined score
    word_df["combined_word"] = (
        word_df["ccs_word"] + word_df["lexical_word"])

    # ── Aggregate to segment level ────────────────────────────
    pr()
    pr("  Aggregating to segment level...")

    agg_dict = {
        "ccs_word":      ["mean", "std", "min", "max"],
        "lexical_word":  ["mean"],
        "combined_word": ["mean"],
    }

    # Add raw feature means for reference
    for feat in CCS_WEIGHTS.keys():
        z_name = z_cols.get(feat)
        if z_name and z_name in word_df.columns:
            agg_dict[z_name] = ["mean"]

    # Word count per segment
    agg_dict["word_resolved"] = ["count"]

    seg_ccs = word_df.groupby("segment_id").agg(agg_dict)
    seg_ccs.columns = [
        f"{col}_{stat}" if stat != "mean" else col
        for col, stat in seg_ccs.columns
    ]
    seg_ccs = seg_ccs.rename(columns={
        "word_resolved_count": "word_count",
        "ccs_word": "ccs_mean",
        "ccs_word_std": "ccs_std",
        "ccs_word_min": "ccs_min",
        "ccs_word_max": "ccs_max",
        "lexical_word": "lexical_mean",
        "combined_word": "combined_mean",
    })
    seg_ccs = seg_ccs.reset_index()

    # ── Add scenario info ─────────────────────────────────────
    if "scenario_corrected" in word_df.columns:
        sc_map = word_df.groupby("segment_id")[
            "scenario_corrected"].first()
        seg_ccs["scenario"] = seg_ccs["segment_id"].map(sc_map)
    elif "scenario" in word_df.columns:
        sc_map = word_df.groupby("segment_id")["scenario"].first()
        seg_ccs["scenario"] = seg_ccs["segment_id"].map(sc_map)

    pr(f"  Segments with CCS: {len(seg_ccs):,}")

    # ── Statistics per scenario ───────────────────────────────
    pr()
    pr("  CCS Reference Distribution:")
    pr(f"  {'Scenario':<16} {'N':>8} {'CCS mean':>10} "
       f"{'CCS sd':>10} {'CCS min':>10} {'CCS max':>10}")
    pr(f"  {'─'*66}")

    stats_lines = []
    for sc in ["All", "Conversation", "Extempore", "Read"]:
        if sc == "All":
            subset = seg_ccs
        else:
            subset = seg_ccs[seg_ccs["scenario"] == sc]
        if len(subset) == 0:
            continue
        n    = len(subset)
        mu   = subset["ccs_mean"].mean()
        sd   = subset["ccs_mean"].std()
        mn   = subset["ccs_mean"].min()
        mx   = subset["ccs_mean"].max()
        pr(f"  {sc:<16} {n:>8,} {mu:>10.4f} "
           f"{sd:>10.4f} {mn:>10.4f} {mx:>10.4f}")
        stats_lines.append(
            f"{sc}: n={n}, mean={mu:.4f}, sd={sd:.4f}, "
            f"min={mn:.4f}, max={mx:.4f}")

    # ── Percentiles (for naturalness scoring) ─────────────────
    pr()
    pr("  CCS Percentiles (All segments):")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        val = seg_ccs["ccs_mean"].quantile(pct / 100)
        pr(f"    P{pct:<3}: {val:.4f}")
        stats_lines.append(f"P{pct}: {val:.4f}")

    # ── Save ──────────────────────────────────────────────────
    seg_ccs.to_csv(CCS_REF_CSV, index=False, encoding="utf-8-sig")
    pr(f"\n  Saved: {CCS_REF_CSV}")
    pr(f"  ({len(seg_ccs):,} rows)")

    with open(CCS_STATS, "w", encoding="utf-8") as f:
        f.write("CCS Reference Distribution Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"CCS Weights (from LMM M4):\n")
        for feat, w in CCS_WEIGHTS.items():
            f.write(f"  {feat}: {w:+.4f}\n")
        f.write(f"\nLexical Weights (from LMM M4):\n")
        for feat, w in LEXICAL_WEIGHTS.items():
            f.write(f"  {feat}: {w:+.4f}\n")
        f.write(f"\n")
        for line in stats_lines:
            f.write(line + "\n")
    pr(f"  Saved: {CCS_STATS}")

    # ── Plot ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: CCS by scenario
        scenarios = ["Conversation", "Extempore", "Read"]
        colors    = ["#1F4E79", "#2E75B6", "#70AD47"]

        for sc, color in zip(scenarios, colors):
            subset = seg_ccs[seg_ccs["scenario"] == sc]
            if len(subset) > 0:
                axes[0].hist(subset["ccs_mean"], bins=80,
                            alpha=0.6, color=color, label=sc,
                            density=True)
        axes[0].set_xlabel("CCS (mean per segment)", fontsize=11)
        axes[0].set_ylabel("Density", fontsize=11)
        axes[0].set_title(
            "Cognitive Constraint Score Distribution\n"
            "by Speech Register (Pipeline V2, n=406K words)",
            fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].axvline(0, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.5)
        axes[0].set_facecolor("#F8F9FA")

        # Right: CCS vs word count
        sample = seg_ccs.sample(
            min(5000, len(seg_ccs)), random_state=42)
        axes[1].scatter(sample["word_count"],
                       sample["ccs_mean"],
                       alpha=0.15, s=3, color="#1F4E79")
        axes[1].set_xlabel("Words per segment", fontsize=11)
        axes[1].set_ylabel("CCS (mean)", fontsize=11)
        axes[1].set_title(
            "CCS vs Segment Length\n"
            "(longer segments converge to corpus mean)",
            fontsize=11, fontweight="bold")
        axes[1].axhline(0, color="red", linewidth=0.8,
                       linestyle="--", alpha=0.5)
        axes[1].set_facecolor("#F8F9FA")

        fig.tight_layout()
        fig.savefig(CCS_FIG, dpi=150, bbox_inches="tight")
        plt.close()
        pr(f"  Saved: {CCS_FIG}")
    except Exception as e:
        pr(f"  Plot error: {e}")

    return seg_ccs


# ══════════════════════════════════════════════════════════════
#  PART 2: CONTEXT-RESPONSE PAIR EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_conversation_pairs(seg_df, ccs_df=None):
    """
    Extract context-response pairs from conversation segments.

    A conversation file has multiple segments from different speakers
    taking turns. We extract:
      Context  = all segments up to turn N (concatenated text)
      Response = segment at turn N+1

    Filters:
      - scenario = "Conversation" (multi-speaker turns)
      - Both context and response must be MFA-ready
      - Response must have >= 3 words
      - Context must have >= 1 prior turn
      - dataset_tier >= 2
    """
    pr()
    pr("=" * 65)
    pr("  PART 2: CONTEXT-RESPONSE PAIR EXTRACTION")
    pr("  From IndicVoices Conversation segments")
    pr("=" * 65)

    # ── Identify scenario column ──────────────────────────────
    if "scenario_corrected" in seg_df.columns:
        seg_df["scenario"] = seg_df["scenario_corrected"].fillna(
            seg_df["scenario"])

    # ── Filter to conversations only ──────────────────────────
    conv_df = seg_df.copy()

    # Quality filter
    if "dataset_tier" in conv_df.columns:
        conv_df["dataset_tier"] = pd.to_numeric(
            conv_df["dataset_tier"], errors="coerce")
        conv_df = conv_df[conv_df["dataset_tier"] >= 2].copy()

    # Scenario filter — conversation only
    conv_scenarios = conv_df[
        conv_df["scenario"].str.lower() == "conversation"]
    pr(f"  Total segments: {len(seg_df):,}")
    pr(f"  Conversation (tier >= 2): {len(conv_scenarios):,}")

    if len(conv_scenarios) == 0:
        pr("  ⚠ No conversation segments found.")
        pr("  Trying all scenarios instead...")
        conv_scenarios = conv_df.copy()
        pr(f"  All scenarios (tier >= 2): {len(conv_scenarios):,}")

    # ── Identify text column ──────────────────────────────────
    text_col = None
    for col in ["clean_hindi_text", "normalized_text",
                "verbatim_text"]:
        if col in conv_scenarios.columns:
            text_col = col
            break
    if text_col is None:
        pr("  ERROR: No text column found")
        return None
    pr(f"  Using text column: {text_col}")

    # ── MFA-ready filter ──────────────────────────────────────
    if "is_mfa_ready" in conv_scenarios.columns:
        conv_scenarios = conv_scenarios[
            conv_scenarios["is_mfa_ready"].astype(str).str.lower()
            == "true"
        ].copy()
        pr(f"  After MFA-ready filter: {len(conv_scenarios):,}")

    # ── Sort by file and segment index ────────────────────────
    # Determine the file grouping column
    file_col = None
    for col in ["file_id", "json_file", "source_file"]:
        if col in conv_scenarios.columns:
            file_col = col
            break
    if file_col is None:
        pr("  ERROR: No file_id column found")
        return None

    # Determine segment order column
    idx_col = None
    for col in ["segment_index", "segment_idx", "seg_index"]:
        if col in conv_scenarios.columns:
            idx_col = col
            break
    if idx_col is None:
        # Try to infer from segment_id
        pr("  No segment_index column — using segment_id order")
        idx_col = "segment_id"

    conv_scenarios[idx_col] = pd.to_numeric(
        conv_scenarios[idx_col], errors="coerce")
    conv_scenarios = conv_scenarios.sort_values(
        [file_col, idx_col])

    # ── Word count filter ─────────────────────────────────────
    conv_scenarios["_word_count"] = conv_scenarios[text_col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0)

    # ── Extract pairs ─────────────────────────────────────────
    pr()
    pr("  Extracting context-response pairs...")

    pairs = []
    file_groups = conv_scenarios.groupby(file_col)

    for file_id, group in file_groups:
        group = group.sort_values(idx_col).reset_index(drop=True)

        if len(group) < 2:
            continue  # Need at least 2 segments for a pair

        for i in range(1, len(group)):
            response_row = group.iloc[i]
            response_text = str(response_row[text_col])
            response_words = len(response_text.split())

            # Response must have >= 3 words
            if response_words < 3:
                continue

            # Context = all prior segments concatenated
            context_rows = group.iloc[:i]
            context_texts = [
                str(r[text_col]) for _, r in context_rows.iterrows()
                if pd.notna(r[text_col]) and len(str(r[text_col]).split()) > 0
            ]

            if len(context_texts) == 0:
                continue

            context = " | ".join(context_texts)
            context_turns = len(context_texts)
            context_words = sum(len(t.split()) for t in context_texts)

            pair = {
                "pair_id": f"pair_{len(pairs):05d}",
                "file_id": file_id,
                "context": context,
                "context_turns": context_turns,
                "context_words": context_words,
                "response": response_text,
                "response_words": response_words,
                "response_segment_id": response_row.get(
                    "segment_id", ""),
                "scenario": response_row.get("scenario", ""),
            }

            # Add speaker info if available
            for col in ["speaker_id", "speaker_id_file",
                        "gender"]:
                if col in response_row.index:
                    pair[f"response_{col}"] = response_row[col]

            # Add CCS of actual response if available
            if ccs_df is not None:
                seg_id = response_row.get("segment_id", "")
                ccs_row = ccs_df[ccs_df["segment_id"] == seg_id]
                if len(ccs_row) > 0:
                    pair["response_ccs"] = ccs_row.iloc[0].get(
                        "ccs_mean", None)

            pairs.append(pair)

    pairs_df = pd.DataFrame(pairs)
    pr(f"  Extracted pairs: {len(pairs_df):,}")

    if len(pairs_df) == 0:
        pr("  ⚠ No pairs extracted. Check data format.")
        return None

    # ── Statistics ─────────────────────────────────────────────
    pr()
    pr("  Pair Statistics:")
    pr(f"  {'Total pairs':<30}: {len(pairs_df):,}")
    pr(f"  {'Unique files':<30}: {pairs_df['file_id'].nunique():,}")
    pr(f"  {'Context turns (mean)':<30}: "
       f"{pairs_df['context_turns'].mean():.1f}")
    pr(f"  {'Context turns (max)':<30}: "
       f"{pairs_df['context_turns'].max()}")
    pr(f"  {'Context words (mean)':<30}: "
       f"{pairs_df['context_words'].mean():.1f}")
    pr(f"  {'Response words (mean)':<30}: "
       f"{pairs_df['response_words'].mean():.1f}")

    if "scenario" in pairs_df.columns:
        pr()
        pr("  By scenario:")
        for sc in pairs_df["scenario"].unique():
            n = (pairs_df["scenario"] == sc).sum()
            pr(f"    {sc}: {n:,}")

    if "response_ccs" in pairs_df.columns:
        valid_ccs = pairs_df["response_ccs"].dropna()
        if len(valid_ccs) > 0:
            pr()
            pr(f"  Response CCS (actual human):")
            pr(f"    mean = {valid_ccs.mean():.4f}")
            pr(f"    sd   = {valid_ccs.std():.4f}")

    # ── Save ──────────────────────────────────────────────────
    pairs_df.to_csv(PAIRS_CSV, index=False, encoding="utf-8-sig")
    pr(f"\n  Saved: {PAIRS_CSV}")
    pr(f"  ({len(pairs_df):,} pairs)")

    # Save stats
    with open(PAIRS_STATS, "w", encoding="utf-8") as f:
        f.write("Context-Response Pair Extraction Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total pairs: {len(pairs_df)}\n")
        f.write(f"Unique files: {pairs_df['file_id'].nunique()}\n")
        f.write(f"Context turns mean: "
                f"{pairs_df['context_turns'].mean():.1f}\n")
        f.write(f"Response words mean: "
                f"{pairs_df['response_words'].mean():.1f}\n")
    pr(f"  Saved: {PAIRS_STATS}")

    return pairs_df


# ══════════════════════════════════════════════════════════════
#  PART 3: SCORING FUNCTION (for future use in Module 3)
# ══════════════════════════════════════════════════════════════

def score_text_ccs(text, stanza_pipeline=None, gpt2_model=None,
                   reference_mean=0.0, reference_sd=1.0):
    """
    Score a single text using CCS.
    This function will be called by Module 3 (cognitive_scorer.py)
    for each LLM candidate response.

    For now, this is a placeholder showing the API.
    Full implementation requires Stanza + GPT-2 Hindi loaded.

    Parameters
    ----------
    text : str
        Hindi text to score
    stanza_pipeline : stanza.Pipeline or None
        Loaded Stanza Hindi pipeline
    gpt2_model : tuple or None
        (model, tokenizer) for surajp/gpt2-hindi
    reference_mean : float
        Mean CCS from reference distribution
    reference_sd : float
        SD of CCS from reference distribution

    Returns
    -------
    dict with keys:
        ccs : float — raw cognitive constraint score
        distance : float — distance from human reference
        naturalness : float — 0-1 naturalness score
        features : dict — individual feature values
    """
    # Placeholder — will be implemented when integrated
    # with Stanza and GPT-2 in Module 3
    return {
        "ccs": 0.0,
        "distance": 0.0,
        "naturalness": 0.5,
        "features": {
            "IC_mean": 0.0,
            "SC_mean": 0.0,
            "fwd_surprisal_mean": 0.0,
            "bwd_surprisal_mean": 0.0,
        }
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    pr("=" * 65)
    pr("  MTP-3 STEP 1: FRAMEWORK + PAIR EXTRACTION")
    pr("  Acoustic-Linguistic Modelling of Hindi Speech")
    pr("  Student : Mayur Shriram Gaidhane (M22AIE248)")
    pr("=" * 65)

    start = time.time()

    # ── Load word features ────────────────────────────────────
    word_path = WORD_CSV if os.path.exists(WORD_CSV) else WORD_CSV_ALT
    if not os.path.exists(word_path):
        pr(f"  ERROR: word_features not found at:")
        pr(f"    {WORD_CSV}")
        pr(f"    {WORD_CSV_ALT}")
        pr("  Run Pipeline V2 full dataset first.")
        return

    pr(f"\n  Loading {word_path} ...")
    word_df = pd.read_csv(word_path, encoding="utf-8-sig",
                          low_memory=False)
    pr(f"  Word features loaded: {len(word_df):,} rows × "
       f"{len(word_df.columns)} cols")

    # ── Part 1: CCS Reference ────────────────────────────────
    ccs_df = compute_ccs_reference(word_df)

    # ── Load segments ─────────────────────────────────────────
    seg_path = SEG_CSV if os.path.exists(SEG_CSV) else SEG_CSV_ALT
    if not os.path.exists(seg_path):
        pr(f"\n  WARNING: segments.csv not found at:")
        pr(f"    {SEG_CSV}")
        pr(f"    {SEG_CSV_ALT}")
        pr("  Skipping pair extraction.")
        elapsed = time.time() - start
        pr(f"\n  Time: {elapsed:.1f}s")
        save_all()
        return

    pr(f"\n  Loading {seg_path} ...")
    seg_df = pd.read_csv(seg_path, encoding="utf-8-sig",
                         low_memory=False)
    pr(f"  Segments loaded: {len(seg_df):,} rows × "
       f"{len(seg_df.columns)} cols")

    # ── Part 2: Pair Extraction ───────────────────────────────
    pairs_df = extract_conversation_pairs(seg_df, ccs_df)

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start
    pr()
    pr("=" * 65)
    pr("  SUMMARY")
    pr("=" * 65)
    pr(f"  CCS reference segments : {len(ccs_df):,}")
    if pairs_df is not None:
        pr(f"  Conversation pairs     : {len(pairs_df):,}")
    pr(f"  Time                   : {elapsed:.1f}s")
    pr()
    pr("  Next steps:")
    pr("  1. Module 2: Generate candidates (Airavata)")
    pr("     python mtp3_step2_candidates.py")
    pr("  2. Module 3: Score candidates with CCS")
    pr("     python mtp3_step3_scoring.py")
    pr("  3. Module 4: Rank and select responses")
    pr("     python mtp3_step4_selection.py")
    pr("=" * 65)

    save_all()
    pr("  Done.")


def save_all():
    """Save the full report."""
    report_path = os.path.join(MTP3_DIR, "step1_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()