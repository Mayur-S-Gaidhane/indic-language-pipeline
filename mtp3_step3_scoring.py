#!/usr/bin/env python3
"""
mtp3_step3_scoring.py — MTP-3 Step 6+7: Cognitive Scoring + Selection

PURPOSE
-------
Score each LLM candidate using the Cognitive Constraint Score (CCS),
rank them, select the most natural response, and compare with the
actual human response.

This implements Proposal Steps 6 (Cognitive Constraint Modelling)
and Step 7 (Response Selection) together.

SCORING
-------
For each candidate text (no audio needed):
  1. Stanza Hindi → IC, SC, POS, dependency tree
  2. GPT-2 Hindi  → forward_surprisal, backward_surprisal
  3. Simple        → word_length_chars, word_frequency
  4. CCS = Σ(βi × mean_feature_i_z)
  5. Distance from human reference CCS
  6. Rank by naturalness → select top candidate

CCS Weights (from LMM V7 M4 — dep_distance removed per Prof):
  IC           : +0.0018 (ns)
  SC           : +0.0033 (*)
  fwd_surprisal: +0.0006 (ns)
  bwd_surprisal: -0.0011 (ns)

INPUT
-----
  outputs/mtp3/candidates.csv       (from Step 2)
  outputs/mtp3/ccs_reference.csv    (from Step 1)

OUTPUT
------
  outputs/mtp3/scored_candidates.csv
  outputs/mtp3/selected_responses.csv
  outputs/mtp3/step3_report.txt

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MTP3_DIR     = os.path.join(ROOT, "outputs", "mtp3")
CAND_CSV     = os.path.join(MTP3_DIR, "candidates.csv")
CCS_REF_CSV  = os.path.join(MTP3_DIR, "ccs_reference.csv")
SCORED_CSV   = os.path.join(MTP3_DIR, "scored_candidates.csv")
SELECTED_CSV = os.path.join(MTP3_DIR, "selected_responses.csv")
REPORT_TXT   = os.path.join(MTP3_DIR, "step3_report.txt")

os.makedirs(MTP3_DIR, exist_ok=True)

# ── CCS Weights from LMM V7 M4 (dep_distance removed per Prof) ──
CCS_WEIGHTS = {
    "IC":                  +0.0018,
    "SC":                  +0.0033,
    "forward_surprisal":   +0.0006,
    "backward_surprisal":  -0.0011,
}

lines = []
def pr(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ── Stanza Feature Extraction ────────────────────────────────
def extract_stanza_features(text, nlp):
    """
    Extract IC, SC from Hindi text using Stanza.
    Matches Prof's dlt.py logic (verified 100% in verify_ic_sc.py).
    dep_distance removed per Prof's Sept 29 feedback.
    """
    try:
        doc = nlp(text)
        words = []

        for sent in doc.sentences:
            # Build IC: baseline 1 + intervening non-PUNCT,
            # assigned to LATER token (both directions)
            ic_dict = {w.id: 1 for w in sent.words}
            for word in sent.words:
                if word.head == 0:
                    continue
                lo = min(word.id, word.head)
                hi = max(word.id, word.head)
                intervening = sum(
                    1 for w2 in sent.words
                    if lo < w2.id < hi and w2.upos != "PUNCT"
                )
                ic_dict[hi] = ic_dict.get(hi, 1) + intervening
            # Zero out PUNCT
            for w in sent.words:
                if w.upos == "PUNCT":
                    ic_dict[w.id] = 0

            # Build SC: baseline 1 + incomplete deps (d <= i < h)
            valid_ids = {w.id for w in sent.words}
            deps = [
                (w.id, w.head) for w in sent.words
                if w.head != 0 and w.head in valid_ids
            ]

            for word in sent.words:
                i = word.id
                sc = 1 + sum(1 for (d, h) in deps if d <= i < h)
                if word.upos == "PUNCT":
                    sc = 0

                words.append({
                    "word": word.text,
                    "IC": ic_dict.get(i, 1),
                    "SC": sc,
                    "pos": word.upos,
                    "dep_rel": word.deprel,
                    "word_length_chars": len(word.text),
                })

        return words
    except Exception as e:
        return []


# ── Surprisal Extraction ─────────────────────────────────────
def extract_surprisal(text, model, tokenizer, device):
    """
    Extract forward and backward surprisal using GPT-2 Hindi.
    Reuses the same logic as Pipeline V2 Phase 5C.
    """
    import torch

    try:
        # Forward surprisal
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        # Per-token log probabilities
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Forward surprisal: -log P(token | left context)
        fwd_surprisals = []
        for i in range(1, inputs["input_ids"].shape[1]):
            token_id = inputs["input_ids"][0, i]
            lp = log_probs[0, i-1, token_id].item()
            fwd_surprisals.append(-lp / np.log(2))  # bits

        fwd_mean = np.mean(fwd_surprisals) if fwd_surprisals else 0.0

        # Backward surprisal: reverse text
        rev_text = " ".join(text.split()[::-1])
        rev_inputs = tokenizer(rev_text, return_tensors="pt",
                                truncation=True, max_length=512)
        rev_inputs = {k: v.to(device) for k, v in rev_inputs.items()}

        with torch.no_grad():
            rev_outputs = model(**rev_inputs,
                                labels=rev_inputs["input_ids"])

        rev_logits = rev_outputs.logits
        rev_log_probs = torch.log_softmax(rev_logits, dim=-1)

        bwd_surprisals = []
        for i in range(1, rev_inputs["input_ids"].shape[1]):
            token_id = rev_inputs["input_ids"][0, i]
            lp = rev_log_probs[0, i-1, token_id].item()
            bwd_surprisals.append(-lp / np.log(2))

        bwd_mean = np.mean(bwd_surprisals) if bwd_surprisals else 0.0

        return fwd_mean, bwd_mean

    except Exception:
        return 0.0, 0.0


# ── CCS Computation ──────────────────────────────────────────
def compute_ccs(word_features, corpus_stats):
    """
    Compute Cognitive Constraint Score for a list of word features.
    Z-scores are computed against corpus-level statistics.
    """
    if len(word_features) == 0:
        return 0.0, {}

    features = {}
    for feat in CCS_WEIGHTS.keys():
        values = [w[feat] for w in word_features
                  if feat in w and w[feat] is not None]
        if values:
            raw_mean = np.mean(values)
            # Z-score against corpus stats
            corpus_mu = corpus_stats.get(f"{feat}_mean", 0)
            corpus_sd = corpus_stats.get(f"{feat}_sd", 1)
            z = (raw_mean - corpus_mu) / corpus_sd if corpus_sd > 0 else 0
            features[feat] = {"raw": raw_mean, "z": z}
        else:
            features[feat] = {"raw": 0.0, "z": 0.0}

    # CCS = weighted sum of z-scored features
    ccs = sum(
        CCS_WEIGHTS[feat] * features[feat]["z"]
        for feat in CCS_WEIGHTS.keys()
    )

    return ccs, features


def main():
    pr("=" * 65)
    pr("  MTP-3 STEP 3: COGNITIVE SCORING + RESPONSE SELECTION")
    pr("  Steps 6+7 from Research Proposal")
    pr("=" * 65)

    start = time.time()

    # ── Load candidates ───────────────────────────────────────
    if not os.path.exists(CAND_CSV):
        pr(f"  ERROR: {CAND_CSV} not found")
        pr("  Run mtp3_step2_candidates.py first")
        return

    cand_df = pd.read_csv(CAND_CSV, encoding="utf-8-sig")
    pr(f"\n  Loaded {len(cand_df):,} candidates "
       f"({cand_df['pair_id'].nunique()} pairs)")

    # ── Load reference CCS ────────────────────────────────────
    if os.path.exists(CCS_REF_CSV):
        ref_df = pd.read_csv(CCS_REF_CSV, encoding="utf-8-sig")
        ref_mean = ref_df["ccs_mean"].mean()
        ref_sd   = ref_df["ccs_mean"].std()
        pr(f"  Reference CCS: mean={ref_mean:.4f}, sd={ref_sd:.4f}")
    else:
        ref_mean = 0.0
        ref_sd   = 1.0
        pr("  WARNING: No reference CCS — using defaults")

    # ── Load Stanza ───────────────────────────────────────────
    pr("\n  Loading Stanza Hindi pipeline...")
    try:
        import stanza
        stanza.download("hi", processors="tokenize,pos,lemma,depparse",
                        verbose=False)
        nlp = stanza.Pipeline("hi",
                              processors="tokenize,pos,lemma,depparse",
                              verbose=False)
        pr("  Stanza loaded ✓")
    except Exception as e:
        pr(f"  Stanza error: {e}")
        pr("  Install: pip install stanza")
        return

    # ── Load GPT-2 Hindi ──────────────────────────────────────
    pr("  Loading GPT-2 Hindi for surprisal...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        gpt2_name = "surajp/gpt2-hindi"
        gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_name)
        gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_name)
        gpt2_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpt2_model = gpt2_model.to(device)
        pr(f"  GPT-2 Hindi loaded ✓ (device: {device})")
    except Exception as e:
        pr(f"  GPT-2 error: {e}")
        gpt2_model = None
        gpt2_tokenizer = None
        device = "cpu"

    # ── Corpus statistics for z-scoring ───────────────────────
    # From LMM V7 descriptive stats (n=406,164 words)
    # dep_distance removed per Prof's Sept 29 feedback
    corpus_stats = {
        "IC_mean": 2.995,          "IC_sd": 7.215,
        "SC_mean": 2.888,          "SC_sd": 1.778,
        "forward_surprisal_mean": 12.408,
        "forward_surprisal_sd": 8.796,
        "backward_surprisal_mean": 8.675,
        "backward_surprisal_sd": 8.585,
    }

    # ── Score each candidate ──────────────────────────────────
    pr("\n  Scoring candidates...")

    scored_rows = []
    for idx, row in cand_df.iterrows():
        text = str(row["candidate_text"])

        # 1. Stanza features
        word_feats = extract_stanza_features(text, nlp)

        # 2. Surprisal
        fwd_surp, bwd_surp = 0.0, 0.0
        if gpt2_model is not None:
            fwd_surp, bwd_surp = extract_surprisal(
                text, gpt2_model, gpt2_tokenizer, device)

        # Add surprisal to word features
        for wf in word_feats:
            wf["forward_surprisal"] = fwd_surp
            wf["backward_surprisal"] = bwd_surp

        # 3. CCS
        ccs, feat_details = compute_ccs(word_feats, corpus_stats)

        # 4. Distance from reference
        distance = abs(ccs - ref_mean) / ref_sd if ref_sd > 0 else 0
        naturalness = 1.0 / (1.0 + distance)

        scored_rows.append({
            "pair_id": row["pair_id"],
            "candidate_id": row["candidate_id"],
            "candidate_text": text,
            "candidate_words": row.get("candidate_words", len(text.split())),
            "is_human": row.get("is_human", 0),
            "source_model": row.get("source_model", ""),
            "IC_mean": feat_details.get("IC", {}).get("raw", 0),
            "SC_mean": feat_details.get("SC", {}).get("raw", 0),
            "fwd_surp_mean": fwd_surp,
            "bwd_surp_mean": bwd_surp,
            "IC_z": feat_details.get("IC", {}).get("z", 0),
            "SC_z": feat_details.get("SC", {}).get("z", 0),
            "fwd_surp_z": (fwd_surp - corpus_stats["forward_surprisal_mean"])
                          / corpus_stats["forward_surprisal_sd"],
            "bwd_surp_z": (bwd_surp - corpus_stats["backward_surprisal_mean"])
                          / corpus_stats["backward_surprisal_sd"],
            "ccs": ccs,
            "distance_from_ref": distance,
            "naturalness_score": naturalness,
        })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start
            pr(f"    Scored {idx+1}/{len(cand_df)} | {elapsed:.0f}s")

    scored_df = pd.DataFrame(scored_rows)
    scored_df.to_csv(SCORED_CSV, index=False, encoding="utf-8-sig")
    pr(f"\n  Saved: {SCORED_CSV} ({len(scored_df):,} rows)")

    # ══════════════════════════════════════════════════════════
    #  RESPONSE SELECTION (Step 7)
    # ══════════════════════════════════════════════════════════
    pr()
    pr("=" * 65)
    pr("  RESPONSE SELECTION (Step 7)")
    pr("  Selecting top-ranked candidate per pair")
    pr("=" * 65)

    selections = []
    human_ranks = []

    for pair_id, group in scored_df.groupby("pair_id"):
        # Separate LLM candidates and human response
        llm_cands = group[group["is_human"] == 0].copy()
        human_row = group[group["is_human"] == 1]

        if len(llm_cands) == 0:
            continue

        # Rank by naturalness score (higher = better)
        llm_cands = llm_cands.sort_values(
            "naturalness_score", ascending=False)

        selected = llm_cands.iloc[0]

        # Where does human response rank among ALL candidates?
        all_ranked = group.sort_values(
            "naturalness_score", ascending=False).reset_index()
        human_rank = None
        if len(human_row) > 0:
            human_cid = human_row.iloc[0]["candidate_id"]
            for rank_idx, r in all_ranked.iterrows():
                if r["candidate_id"] == human_cid:
                    human_rank = rank_idx + 1
                    break
            if human_rank:
                human_ranks.append(human_rank)

        selections.append({
            "pair_id": pair_id,
            "selected_text": selected["candidate_text"],
            "selected_ccs": selected["ccs"],
            "selected_naturalness": selected["naturalness_score"],
            "human_text": (human_row.iloc[0]["candidate_text"]
                          if len(human_row) > 0 else ""),
            "human_ccs": (human_row.iloc[0]["ccs"]
                         if len(human_row) > 0 else None),
            "human_naturalness": (human_row.iloc[0]["naturalness_score"]
                                 if len(human_row) > 0 else None),
            "human_rank": human_rank,
            "n_candidates": len(llm_cands),
            "ccs_distance": abs(
                selected["ccs"] -
                (human_row.iloc[0]["ccs"]
                 if len(human_row) > 0 else 0)),
        })

    sel_df = pd.DataFrame(selections)
    sel_df.to_csv(SELECTED_CSV, index=False, encoding="utf-8-sig")

    # ── Summary Statistics ────────────────────────────────────
    pr()
    pr("  Selection Results:")
    pr(f"  Pairs with selection   : {len(sel_df):,}")
    pr(f"  Avg naturalness (sel)  : "
       f"{sel_df['selected_naturalness'].mean():.4f}")
    if "human_naturalness" in sel_df:
        hnat = sel_df["human_naturalness"].dropna()
        if len(hnat) > 0:
            pr(f"  Avg naturalness (human): {hnat.mean():.4f}")

    if human_ranks:
        mrr = np.mean([1.0/r for r in human_ranks])
        pr(f"  Human response MRR     : {mrr:.4f}")
        pr(f"  Human avg rank         : {np.mean(human_ranks):.1f}")
        top1 = sum(1 for r in human_ranks if r == 1)
        pr(f"  Human in top-1         : {top1}/{len(human_ranks)} "
           f"({100*top1/len(human_ranks):.1f}%)")

    pr(f"\n  Saved: {SELECTED_CSV}")

    # ── Show sample ───────────────────────────────────────────
    pr()
    pr("  Sample selection (first pair):")
    if len(sel_df) > 0:
        s = sel_df.iloc[0]
        pr(f"    Selected: {str(s['selected_text'])[:80]}")
        pr(f"    CCS     : {s['selected_ccs']:.4f} "
           f"(naturalness: {s['selected_naturalness']:.4f})")
        pr(f"    Human   : {str(s['human_text'])[:80]}")
        if s['human_ccs'] is not None:
            pr(f"    H.CCS   : {s['human_ccs']:.4f} "
               f"(rank: {s['human_rank']})")

    elapsed = time.time() - start
    pr()
    pr(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    pr()
    pr("  Next steps:")
    pr("  - Step 8: TTS synthesis (mtp3_step4_tts.py)")
    pr("  - Evaluation: mtp3_step5_evaluate.py")
    pr("=" * 65)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pr(f"  Report saved: {REPORT_TXT}")


if __name__ == "__main__":
    main()