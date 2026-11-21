#!/usr/bin/env python3
"""
hdtb_pipeline.py — CCS Pipeline on UD Hindi-HDTB (Written Text)
Applies the same CCS response selection logic used for Gadkari interview
to the Hindi Dependency Treebank (news/written text domain).

Student  : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur

Pipeline:
  1. Parse CoNLL-U files (gold dependency parses)
  2. Extract sequential sentence pairs (context → next sentence)
  3. Compute CCS reference from train split
  4. Generate LLM candidates (Sarvam + Gemini) with V5 prompt
  5. CCS scoring + naturalness ranking + response selection
  6. Evaluate: MRR, BLEU, ROUGE, BERTScore (raw + scaled), Embedding Cosine Sim
  7. Bonus: Compare gold IC/SC vs Stanza-parsed IC/SC

Usage:
    export SARVAM_API_KEY='your_key'
    export GEMINI_API_KEY='your_key'

    python hdtb_pipeline.py                                # default: 50 test pairs, context=3, V5
    python hdtb_pipeline.py --n_pairs 100 --context 5      # 100 pairs, 5-sentence context
    python hdtb_pipeline.py --prompt_version 4              # V4 prompt
    python hdtb_pipeline.py --skip_generation               # re-score existing candidates
    python hdtb_pipeline.py --gold_vs_stanza                # compare gold vs Stanza IC/SC

Output:
    outputs/hdtb_eval/  (all results stored here)
"""

import os, sys, json, time, re, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Parse arguments ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="CCS Pipeline — UD Hindi-HDTB Evaluation")
parser.add_argument("--context", type=int, default=3, help="Number of context sentences (default: 3)")
parser.add_argument("--n_pairs", type=int, default=50, help="Number of test pairs to evaluate (default: 50)")
parser.add_argument("--prompt_version", type=int, default=5, choices=[4, 5], help="Prompt version (4 or 5)")
parser.add_argument("--n_sarvam", type=int, default=10, help="Sarvam candidates per pair")
parser.add_argument("--n_gemini", type=int, default=10, help="Gemini candidates per pair")
parser.add_argument("--skip_generation", action="store_true", help="Skip LLM generation, re-score existing candidates")
parser.add_argument("--gold_vs_stanza", action="store_true", help="Run gold vs Stanza IC/SC comparison")
parser.add_argument("--precomputed_ccs", action="store_true",
                    help="Skip CCS reference computation (Step 3), use precomputed values from train")
parser.add_argument("--seed", type=int, default=42, help="Random seed for pair sampling")
parser.add_argument("--pair_ids", type=str, default=None,
                    help="Comma-separated pair IDs to evaluate (e.g. 'hdtb_test_100,hdtb_test_500'). "
                         "Overrides --n_pairs and --seed. Check all_test_pairs_dataset.csv for available IDs.")
args = parser.parse_args()

# ── Precomputed Constants ──────────────────────────────────
#
# These constants are precomputed from the HDTB train split to save time
# on subsequent runs. Use --precomputed_ccs to skip Step 3.
#
# HOW THESE VALUES WERE COMPUTED:
#
# 1. CCS Reference (Mean & SD):
#    - Source: hi_hdtb-ud-train.conllu (13,306 sentences)
#    - Method: Random sample of 2000 sentences (seed=42)
#    - For each sentence: gold IC/SC from CoNLL-U dependency parse +
#      forward/backward surprisal from surajp/gpt2-hindi
#    - CCS = W_IC*IC + W_SC*SC + W_FWD*fwd + W_BWD*bwd
#      (weights from LMM V7 M4: W_IC=+0.0018, W_SC=+0.0033, W_FWD=+0.0006, W_BWD=-0.0011)
#    - First computed: 2026-11-02, hdtb_pipeline.py test run
#
# 2. BERTScore Baseline:
#    - Source: HDTB train split, 500 random sentence pairs (seed=42)
#    - Model: bert-base-multilingual-cased (same as evaluation)
#    - Method: BERTScore F1 between random (non-adjacent) sentence pairs
#      This gives the expected BERTScore when comparing unrelated Hindi
#      news sentences — used to scale raw BERTScore into [baseline, 1] range.
#    - NOTE: This is DIFFERENT from the Gadkari baseline (0.6499) which was
#      computed on conversational speech. Written news text has different
#      vocabulary distribution, so domain-specific baseline is needed.
#    - Value: computed on first run, then hardcoded below.
#    - First computed: <pending first full run>
#
PRECOMPUTED_CCS_MEAN = 0.016379
PRECOMPUTED_CCS_SD   = 0.004857
# Computed on 2026-11-02: HDTB train, 500 random non-adjacent pairs (seed=42),
# bert-base-multilingual-cased. Higher than Gadkari's 0.6499 because news text
# sentences share more vocabulary/structure than conversational speech.
PRECOMPUTED_BERTSCORE_BASELINE_HDTB = 0.6640

# ── Paths ───────────────────────────────────────────────────
HDTB_DIR   = os.path.join(ROOT, "hdtb_data")
TRAIN_FILE = os.path.join(HDTB_DIR, "hi_hdtb-ud-train.conllu")
DEV_FILE   = os.path.join(HDTB_DIR, "hi_hdtb-ud-dev.conllu")
TEST_FILE  = os.path.join(HDTB_DIR, "hi_hdtb-ud-test.conllu")

OUT_DIR = os.path.join(ROOT, "outputs", f"hdtb_eval_v{args.prompt_version}_ctx{args.context}")
os.makedirs(OUT_DIR, exist_ok=True)

CAND_CSV   = os.path.join(OUT_DIR, "candidates.csv")
SCORED_CSV = os.path.join(OUT_DIR, "scored_candidates.csv")
SELECT_CSV = os.path.join(OUT_DIR, "selected_responses.csv")
REPORT_TXT = os.path.join(OUT_DIR, "hdtb_eval_report.txt")

# ── Logging ─────────────────────────────────────────────────
_log_lines = []
def pr(msg=""):
    print(msg)
    _log_lines.append(msg)

t0 = time.time()

pr("=" * 65)
pr(f"  HDTB PIPELINE: CCS Evaluation on UD Hindi-HDTB")
pr("=" * 65)
pr(f"  Context window  : {args.context} sentences")
pr(f"  Test pairs      : {args.n_pairs}")
pr(f"  Prompt version  : V{args.prompt_version}")
pr(f"  Output          : {OUT_DIR}")
pr()

# ══════════════════════════════════════════════════════════════
#  STEP 1: Parse CoNLL-U Files
# ══════════════════════════════════════════════════════════════
pr(f"{'─'*65}")
pr("  STEP 1: Parse CoNLL-U Files")
pr(f"{'─'*65}")

def parse_conllu(filepath):
    """Parse a CoNLL-U file and return list of sentence dicts.
    Each sentence dict has:
      - sent_id: str
      - text: str (the full sentence text)
      - tokens: list of dicts with id, form, lemma, upos, xpos, feats, head, deprel
    """
    sentences = []
    current_tokens = []
    sent_id = None
    sent_text = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text"):
                sent_text = line.split("=", 1)[1].strip()
            elif line == "":
                # End of sentence
                if current_tokens and sent_text:
                    sentences.append({
                        "sent_id": sent_id or f"sent_{len(sentences)+1}",
                        "text": sent_text,
                        "tokens": current_tokens
                    })
                current_tokens = []
                sent_id = None
                sent_text = None
            elif not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 8:
                    tok_id = parts[0]
                    # Skip multi-word tokens (e.g., "1-2") and empty nodes
                    if "-" in tok_id or "." in tok_id:
                        continue
                    try:
                        current_tokens.append({
                            "id": int(tok_id),
                            "form": parts[1],
                            "lemma": parts[2],
                            "upos": parts[3],
                            "xpos": parts[4],
                            "feats": parts[5],
                            "head": int(parts[6]) if parts[6] != "_" else 0,
                            "deprel": parts[7],
                        })
                    except ValueError:
                        continue

        # Handle last sentence if file doesn't end with blank line
        if current_tokens and sent_text:
            sentences.append({
                "sent_id": sent_id or f"sent_{len(sentences)+1}",
                "text": sent_text,
                "tokens": current_tokens
            })

    return sentences

pr(f"\n  Parsing {TRAIN_FILE}...")
train_sents = parse_conllu(TRAIN_FILE)
pr(f"    Train sentences: {len(train_sents)}")

pr(f"  Parsing {TEST_FILE}...")
test_sents = parse_conllu(TEST_FILE)
pr(f"    Test sentences : {len(test_sents)}")

if os.path.exists(DEV_FILE):
    dev_sents = parse_conllu(DEV_FILE)
    pr(f"    Dev sentences  : {len(dev_sents)}")
else:
    dev_sents = []

pr(f"    Total          : {len(train_sents) + len(test_sents) + len(dev_sents)}")

# Show sample sentence
if test_sents:
    s = test_sents[0]
    pr(f"\n  Sample sentence:")
    pr(f"    ID  : {s['sent_id']}")
    pr(f"    Text: {s['text'][:100]}...")
    pr(f"    Tokens: {len(s['tokens'])}")

# ══════════════════════════════════════════════════════════════
#  STEP 2: Compute Gold IC/SC from Dependency Parses
# ══════════════════════════════════════════════════════════════
pr(f"\n{'─'*65}")
pr("  STEP 2: Compute Gold IC/SC from CoNLL-U Dependency Parses")
pr(f"{'─'*65}")

def compute_ic_sc_from_tokens(tokens):
    """Compute IC and SC for each token from gold dependency parse.
    Uses the same DLT logic as phase_B_stanza.py / dlt.py:
      IC = 1 + sum of distances for completed deps (backward-looking)
      SC = 1 + count of incomplete deps (forward-looking)
    Non-PUNCT interveners only.
    """
    n = len(tokens)
    ic_list = []
    sc_list = []

    for i, tok in enumerate(tokens):
        pos_i = tok["id"]  # 1-indexed position
        # --- IC: completed dependencies at this position ---
        # A dep (head, dependent) completes at the LATER of head/dep positions
        ic = 1  # minimum cost
        for t in tokens:
            dep_pos = t["id"]
            head_pos = t["head"]
            if head_pos == 0:
                continue  # root
            # This dep completes at max(dep_pos, head_pos)
            completion_pos = max(dep_pos, head_pos)
            if completion_pos == pos_i:
                # Count non-PUNCT interveners between dep and head
                lo = min(dep_pos, head_pos)
                hi = max(dep_pos, head_pos)
                interveners = 0
                for k in tokens:
                    if lo < k["id"] < hi and k["upos"] != "PUNCT":
                        interveners += 1
                ic += interveners

        # --- SC: incomplete dependencies at this position ---
        sc = 1  # minimum cost
        for t in tokens:
            dep_pos = t["id"]
            head_pos = t["head"]
            if head_pos == 0:
                continue
            lo = min(dep_pos, head_pos)
            hi = max(dep_pos, head_pos)
            # A dep is "open" at pos_i if lo <= pos_i < hi
            if lo <= pos_i < hi:
                sc += 1

        # PUNCT tokens get IC=0, SC=0 (matching eval script logic)
        if tok["upos"] == "PUNCT":
            ic_list.append(0)
            sc_list.append(0)
        else:
            ic_list.append(ic)
            sc_list.append(sc)

    return ic_list, sc_list

def compute_sentence_ccs_features(tokens):
    """Compute mean IC, SC for a sentence from gold parse."""
    if not tokens:
        return {"mean_ic": 0, "mean_sc": 0, "n_tokens": 0}
    ic_list, sc_list = compute_ic_sc_from_tokens(tokens)
    return {
        "mean_ic": np.mean(ic_list),
        "mean_sc": np.mean(sc_list),
        "n_tokens": len(tokens),
    }

# Compute gold IC/SC for all train sentences (for CCS reference)
pr(f"\n  Computing gold IC/SC for {len(train_sents)} train sentences...")
train_ccs_data = []
for i, sent in enumerate(train_sents):
    feats = compute_sentence_ccs_features(sent["tokens"])
    train_ccs_data.append(feats)
    if (i + 1) % 2000 == 0:
        pr(f"    Processed {i+1}/{len(train_sents)}...")

pr(f"    Done. Computed IC/SC for {len(train_ccs_data)} sentences.")

# ══════════════════════════════════════════════════════════════
#  STEP 3: Compute CCS Reference from Train + Surprisal
# ══════════════════════════════════════════════════════════════
pr(f"\n{'─'*65}")
pr("  STEP 3: CCS Reference from Train (IC + SC + Surprisal)")
pr(f"{'─'*65}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# CCS weights from LMM V7 M4 (same as Gadkari pipeline)
W_IC  = +0.0018
W_SC  = +0.0033
W_FWD = +0.0006
W_BWD = -0.0011

# Corpus statistics for z-scoring (from training corpus analysis)
CORPUS_STATS = {
    "IC_mean": 2.995, "IC_sd": 7.215,
    "SC_mean": 4.047, "SC_sd": 3.571,
    "fwd_surp_mean": 7.871, "fwd_surp_sd": 3.300,
    "bwd_surp_mean": 9.882, "bwd_surp_sd": 2.819,
}

def compute_ccs_score(mean_ic, mean_sc, fwd_surp, bwd_surp):
    """CCS = weighted sum of z-scored features against corpus stats."""
    ic_z  = (mean_ic  - CORPUS_STATS["IC_mean"])  / CORPUS_STATS["IC_sd"]
    sc_z  = (mean_sc  - CORPUS_STATS["SC_mean"])  / CORPUS_STATS["SC_sd"]
    fwd_z = (fwd_surp - CORPUS_STATS["fwd_surp_mean"]) / CORPUS_STATS["fwd_surp_sd"]
    bwd_z = (bwd_surp - CORPUS_STATS["bwd_surp_mean"]) / CORPUS_STATS["bwd_surp_sd"]
    return W_IC * ic_z + W_SC * sc_z + W_FWD * fwd_z + W_BWD * bwd_z

def compute_surprisal(text, tokenizer, model, dev):
    """Compute mean forward and backward surprisal for text."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(dev)
    if input_ids.shape[1] < 2:
        return 0.0, 0.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Forward surprisal: mean negative log-likelihood
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    fwd_surprisal = -token_log_probs.mean().item() / np.log(2)  # nats → bits

    # Backward surprisal: reverse the text
    rev_text = " ".join(text.split()[::-1])
    enc_rev = tokenizer(rev_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids_rev = enc_rev["input_ids"].to(dev)
    if input_ids_rev.shape[1] < 2:
        return fwd_surprisal, 0.0
    with torch.no_grad():
        outputs_rev = model(input_ids_rev, labels=input_ids_rev)
    logits_rev = outputs_rev.logits[:, :-1, :]
    targets_rev = input_ids_rev[:, 1:]
    log_probs_rev = torch.nn.functional.log_softmax(logits_rev, dim=-1)
    token_log_probs_rev = log_probs_rev.gather(2, targets_rev.unsqueeze(-1)).squeeze(-1)
    bwd_surprisal = -token_log_probs_rev.mean().item() / np.log(2)  # nats → bits

    return fwd_surprisal, bwd_surprisal

if args.precomputed_ccs:
    # ── Use precomputed values (skip GPT-2 loading + 2000 sentence computation) ──
    # See "Precomputed Constants" section at top of file for how these were derived
    ref_mean = PRECOMPUTED_CCS_MEAN
    ref_sd   = PRECOMPUTED_CCS_SD
    pr(f"\n  Using PRECOMPUTED CCS reference (--precomputed_ccs flag):")
    pr(f"    Mean CCS : {ref_mean:.6f}")
    pr(f"    SD CCS   : {ref_sd:.6f}")
    pr(f"    (from HDTB train, 2000 sentences, seed=42, surajp/gpt2-hindi)")
    pr(f"    Skipping GPT-2 loading + surprisal computation — saves ~3-4 min")

    # Still need GPT-2 for candidate scoring in Step 6
    pr(f"\n  Loading GPT-2 Hindi for candidate scoring...")
    gpt2_name = "surajp/gpt2-hindi"
    gpt2_tok = AutoTokenizer.from_pretrained(gpt2_name)
    gpt2_mdl = AutoModelForCausalLM.from_pretrained(gpt2_name)
    gpt2_mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2_mdl.to(device)
    pr(f"  GPT-2 Hindi loaded (device: {device})")

else:
    # ── Compute fresh CCS reference from train split ──
    pr("\n  Loading GPT-2 Hindi for surprisal...")
    gpt2_name = "surajp/gpt2-hindi"
    gpt2_tok = AutoTokenizer.from_pretrained(gpt2_name)
    gpt2_mdl = AutoModelForCausalLM.from_pretrained(gpt2_name)
    gpt2_mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2_mdl.to(device)
    pr(f"  GPT-2 Hindi loaded (device: {device})")

    pr(f"\n  Computing CCS reference from train split...")
    MAX_TRAIN_FOR_REF = 2000  # sample for speed — 2000 sentences enough for stable stats
    rng = np.random.RandomState(args.seed)
    if len(train_sents) > MAX_TRAIN_FOR_REF:
        train_sample_idx = rng.choice(len(train_sents), MAX_TRAIN_FOR_REF, replace=False)
        pr(f"    Sampling {MAX_TRAIN_FOR_REF} of {len(train_sents)} train sentences for CCS reference")
    else:
        train_sample_idx = list(range(len(train_sents)))

    train_ccs_scores = []
    for count, idx in enumerate(train_sample_idx):
        sent = train_sents[idx]
        feats = train_ccs_data[idx]
        fwd, bwd = compute_surprisal(sent["text"], gpt2_tok, gpt2_mdl, device)
        ccs = compute_ccs_score(feats["mean_ic"], feats["mean_sc"], fwd, bwd)
        train_ccs_scores.append(ccs)
        if (count + 1) % 500 == 0:
            pr(f"    CCS computed: {count+1}/{len(train_sample_idx)}")

    ref_mean = np.mean(train_ccs_scores)
    ref_sd = np.std(train_ccs_scores)
    pr(f"\n  CCS Reference (HDTB train, {len(train_ccs_scores)} sentences):")
    pr(f"    Mean CCS : {ref_mean:.6f}")
    pr(f"    SD CCS   : {ref_sd:.6f}")
    pr(f"    *** If these match precomputed values, use --precomputed_ccs next time ***")

# ══════════════════════════════════════════════════════════════
#  STEP 4: Extract Test Pairs (Context → Next Sentence)
# ══════════════════════════════════════════════════════════════
pr(f"\n{'─'*65}")
pr(f"  STEP 4: Extract Test Pairs ({args.context}-sentence context)")
pr(f"{'─'*65}")

def extract_pairs(sentences, context_size):
    """Extract pairs: context_size consecutive sentences → next sentence.
    Each pair: { pair_id, context_texts, target_text, target_tokens }
    """
    pairs = []
    for i in range(context_size, len(sentences)):
        context_texts = [sentences[j]["text"] for j in range(i - context_size, i)]
        target = sentences[i]
        pairs.append({
            "pair_id": f"hdtb_test_{i}",
            "context_texts": context_texts,
            "context_joined": " ".join(context_texts),
            "target_text": target["text"],
            "target_tokens": target["tokens"],
            "target_sent_id": target["sent_id"],
        })
    return pairs

all_test_pairs = extract_pairs(test_sents, args.context)
pr(f"  Total possible test pairs: {len(all_test_pairs)}")

# Ensure rng is always defined for pair sampling
rng = np.random.RandomState(args.seed)

# Select pairs for evaluation
if args.pair_ids:
    # User specified exact pair IDs — override n_pairs and seed
    requested_ids = [pid.strip() for pid in args.pair_ids.split(",")]
    pair_id_map = {p["pair_id"]: p for p in all_test_pairs}
    test_pairs = []
    for pid in requested_ids:
        if pid in pair_id_map:
            test_pairs.append(pair_id_map[pid])
        else:
            pr(f"  [WARN] Pair ID '{pid}' not found — skipping")
    if not test_pairs:
        pr("  ERROR: No valid pair IDs found. Check all_test_pairs_dataset.csv for available IDs.")
        sys.exit(1)
    pr(f"  Selected {len(test_pairs)} pairs by ID: {', '.join(p['pair_id'] for p in test_pairs)}")
elif args.n_pairs < len(all_test_pairs):
    sample_idx = rng.choice(len(all_test_pairs), args.n_pairs, replace=False)
    sample_idx.sort()
    test_pairs = [all_test_pairs[i] for i in sample_idx]
    pr(f"  Sampled {args.n_pairs} pairs for evaluation (seed={args.seed})")
else:
    test_pairs = all_test_pairs
    pr(f"  Using all {len(test_pairs)} pairs")

# Show sample pair
if test_pairs:
    p = test_pairs[0]
    pr(f"\n  Sample pair ({p['pair_id']}):")
    pr(f"    Context ({args.context} sentences):")
    for j, ct in enumerate(p["context_texts"]):
        pr(f"      [{j+1}] {ct[:80]}...")
    pr(f"    Target : {p['target_text'][:80]}...")
    pr(f"    Target tokens: {len(p['target_tokens'])}")

# Save ALL possible pairs as a dataset CSV (for verification/review)
# One-time creation — skip if file already exists (same test split = same pairs)
ALL_PAIRS_CSV = os.path.join(OUT_DIR, f"all_test_pairs_dataset_ctx{args.context}.csv")
if os.path.exists(ALL_PAIRS_CSV):
    pr(f"\n  Pairs dataset already exists: {ALL_PAIRS_CSV} (skipping)")
else:
    pair_rows = []
    for pair in all_test_pairs:
        ctx = pair["context_texts"]
        row = {
            "pair_id": pair["pair_id"],
            "sent_id": pair["target_sent_id"],
            "context_1": ctx[0] if len(ctx) > 0 else "",
            "context_2": ctx[1] if len(ctx) > 1 else "",
            "context_3": ctx[2] if len(ctx) > 2 else "",
        }
        # Add context_4, context_5 if context window > 3
        if args.context >= 4:
            row["context_4"] = ctx[3] if len(ctx) > 3 else ""
        if args.context >= 5:
            row["context_5"] = ctx[4] if len(ctx) > 4 else ""
        row["ground_truth"] = pair["target_text"]
        row["n_tokens"] = len(pair["target_tokens"])
        row["sampled_for_eval"] = pair["pair_id"] in [p["pair_id"] for p in test_pairs]
        pair_rows.append(row)

    pairs_dataset = pd.DataFrame(pair_rows)
    pairs_dataset.to_csv(ALL_PAIRS_CSV, index=False, encoding="utf-8-sig")
    pr(f"\n  Saved ALL {len(all_test_pairs)} pairs dataset: {ALL_PAIRS_CSV}")
    pr(f"    (sampled_for_eval=True marks the {len(test_pairs)} pairs used in this run)")

# ══════════════════════════════════════════════════════════════
#  STEP 5: Generate LLM Candidates
# ══════════════════════════════════════════════════════════════
pr(f"\n{'─'*65}")
pr(f"  STEP 5: Generate LLM Candidates (V{args.prompt_version} Prompt)")
pr(f"{'─'*65}")

# ── V5 Semantic Hints (Stanza-based) ──
if args.prompt_version == 5:
    pr("\n  Loading Stanza Hindi for V5 semantic hints...")
    import stanza
    try:
        nlp_stanza = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                                      use_gpu=torch.cuda.is_available(), verbose=False)
    except Exception:
        stanza.download("hi", verbose=False)
        nlp_stanza = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                                      use_gpu=torch.cuda.is_available(), verbose=False)
    pr("  Stanza Hindi loaded")

    def extract_semantic_hints(text, nlp):
        """Extract keywords, theme, structure from text using Stanza."""
        doc = nlp(text)
        keywords = []
        subjects, verbs_main, objects = [], [], []
        n_sentences = len(doc.sentences)
        n_words = 0
        has_filler = False

        for sent in doc.sentences:
            for word in sent.words:
                n_words += 1
                if word.upos in ("NOUN", "PROPN", "VERB", "ADJ", "NUM"):
                    if word.text not in keywords:
                        keywords.append(word.text)
                if word.deprel in ("nsubj", "nsubj:pass"):
                    subjects.append(word.text)
                if word.upos == "VERB" and word.deprel in ("root", "advcl", "xcomp", "ccomp"):
                    verbs_main.append(word.text)
                if word.deprel in ("obj", "iobj"):
                    objects.append(word.text)
                if word.text in ("हां", "हम्म", "अच्छा", "ठीक", "बस"):
                    has_filler = True

        # Theme classification
        theme = ""
        if subjects and verbs_main:
            svo = f"{subjects[0]} — {verbs_main[0]}"
            if objects:
                svo += f" — {objects[0]}"
            # Simple theme detection
            neg_words = [w.text for s in doc.sentences for w in s.words if w.upos == "PART" and w.text in ("नहीं", "न", "मत", "ना")]
            q_words = [w.text for s in doc.sentences for w in s.words if w.text in ("क्या", "कैसे", "क्यों", "कब", "कहां", "कौन", "किस")]
            if q_words:
                theme = f"प्रश्न ({svo})"
            elif neg_words:
                theme = f"खंडन/असहमति ({svo})"
            else:
                theme = f"कथन ({svo})"
        elif verbs_main:
            theme = f"कथन ({verbs_main[0]})"

        return {
            "keywords": keywords[:8],
            "theme": theme,
            "n_sentences": n_sentences,
            "n_words": n_words,
            "has_filler": has_filler,
        }

# ── Prompt Templates ──

# V4: Minimal prompt for written Hindi (no speaker persona)
SYSTEM_PROMPT_V4 = (
    "आप एक हिंदी लेखक हैं। आपको दिए गए संदर्भ के आधार पर अगला वाक्य लिखना है।\n"
    "कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ अगला वाक्य लिखें।"
)

PROMPT_TEMPLATE_V4 = (
    "संदर्भ:\n{context}\n\n"
    "ऊपर के संदर्भ के आधार पर अगला वाक्य लिखिए।\n"
    "कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ अगला वाक्य दें।\n\n"
    "अगला वाक्य:"
)

# V5: Embedding-guided (semantic hints from ground truth)
SYSTEM_PROMPT_V5 = (
    "आप एक हिंदी लेखक हैं। आपको दिए गए संदर्भ और अर्थ-संकेतों (semantic hints) के आधार पर अगला वाक्य लिखना है।\n"
    "आपको अगले वाक्य का अर्थ (semantic meaning) बताया जा रहा है — मुख्य शब्द, भाव, और संरचना।\n"
    "इन संकेतों के आधार पर स्वाभाविक हिंदी वाक्य लिखिए।\n"
    "कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ वाक्य लिखें।"
)

PROMPT_TEMPLATE_V5 = (
    "संदर्भ:\n{context}\n\n"
    "अगले वाक्य में ये तत्व होने चाहिए:\n"
    "मुख्य शब्द (keywords): {keywords}\n"
    "भाव/विषय (theme): {theme}\n"
    "वाक्य संरचना (structure hint): {structure}\n\n"
    "इन तत्वों के आधार पर लगभग {n_words} शब्दों में स्वाभाविक वाक्य लिखिए।\n"
    "कोई अंग्रेजी नहीं। कोई व्याख्या नहीं। सिर्फ वाक्य दें।\n\n"
    "अगला वाक्य:"
)

def build_prompt(pair, prompt_version, nlp=None):
    """Build system and user prompt for a test pair."""
    context_text = "\n".join(pair["context_texts"])

    if prompt_version == 4:
        system_prompt = SYSTEM_PROMPT_V4
        user_prompt = PROMPT_TEMPLATE_V4.format(context=context_text)
    elif prompt_version == 5:
        hints = extract_semantic_hints(pair["target_text"], nlp)
        kw_str = ", ".join(hints["keywords"]) if hints["keywords"] else "—"
        theme_str = hints["theme"] if hints["theme"] else "—"
        struct_str = f"{hints['n_sentences']} वाक्य, {hints['n_words']} शब्द"
        system_prompt = SYSTEM_PROMPT_V5
        user_prompt = PROMPT_TEMPLATE_V5.format(
            context=context_text,
            keywords=kw_str,
            theme=theme_str,
            structure=struct_str,
            n_words=hints["n_words"],
        )
    else:
        raise ValueError(f"Unknown prompt version: {prompt_version}")

    return system_prompt, user_prompt

# ── LLM Candidate Generation ──

if not args.skip_generation:
    # Sarvam client
    sarvam_client = None
    sarvam_key = os.environ.get("SARVAM_API_KEY", "")
    if sarvam_key:
        try:
            from openai import OpenAI as OAI
            sarvam_client = OAI(
                base_url="https://api.sarvam.ai/v1",
                api_key=sarvam_key,
            )
            pr("  Sarvam AI client loaded")
        except Exception as e:
            pr(f"  [WARN] Sarvam init failed: {e}")

    # Gemini client
    gemini_client = None
    gemini_model = None
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from google import genai as gai
            gemini_client = gai.Client(api_key=gemini_key)
            # Auto-detect available model
            for m in ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-3.5-flash-lite"]:
                try:
                    test_resp = gemini_client.models.generate_content(
                        model=m,
                        contents="नमस्ते",
                        config={"max_output_tokens": 10, "temperature": 0.5},
                    )
                    gemini_model = m
                    pr(f"  Gemini client loaded (model: {m})")
                    break
                except Exception as e:
                    pr(f"    ✗ {m}: {str(e)[:60]}")
            if not gemini_model:
                pr("  [WARN] No Gemini model available")
                gemini_client = None
        except Exception as e:
            pr(f"  [WARN] Gemini init failed: {e}")

    def generate_sarvam(system_prompt, user_prompt, n, temps):
        """Generate candidates via Sarvam API."""
        results = []
        for i, temp in enumerate(temps[:n]):
            try:
                resp = sarvam_client.chat.completions.create(
                    model="sarvam-105b-conversations",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=200,
                    temperature=float(temp),
                    top_p=0.95,
                )
                text = resp.choices[0].message.content.strip()
                # Clean: take first sentence/line
                text = text.split("\n")[0].strip()
                results.append({"text": text, "source": "sarvam", "temp": float(temp)})
            except Exception as e:
                pr(f"    [WARN] Sarvam S{i:02d} failed: {str(e)[:60]}")
            time.sleep(2)  # rate-limit delay (2s for free-tier API limits)
        return results

    def generate_gemini(system_prompt, user_prompt, n, temps):
        """Generate candidates via Gemini API."""
        results = []
        combined = system_prompt + "\n\n" + user_prompt
        for i, temp in enumerate(temps[:n]):
            try:
                resp = gemini_client.models.generate_content(
                    model=gemini_model,
                    contents=combined,
                    config={"max_output_tokens": 200, "temperature": float(temp)},
                )
                text = resp.text.strip().split("\n")[0].strip()
                results.append({"text": text, "source": "gemini", "temp": float(temp)})
            except Exception as e:
                pr(f"    [WARN] Gemini G{i:02d} failed: {str(e)[:60]}")
            time.sleep(2)  # rate-limit delay (2s for free-tier API limits)
        return results

    # Temperature ranges
    if args.prompt_version == 5:
        sarvam_temps = np.linspace(0.3, 1.2, args.n_sarvam)
    else:
        sarvam_temps = np.linspace(0.3, 1.5, args.n_sarvam)
    gemini_temps = np.linspace(0.4, 1.2, args.n_gemini)

    # Generate candidates for all test pairs
    all_rows = []
    pr(f"\n  Generating candidates for {len(test_pairs)} pairs...")

    for pi, pair in enumerate(test_pairs):
        sys_p, usr_p = build_prompt(pair, args.prompt_version,
                                     nlp=nlp_stanza if args.prompt_version == 5 else None)

        candidates = []
        if sarvam_client:
            candidates += generate_sarvam(sys_p, usr_p, args.n_sarvam, sarvam_temps)
        if gemini_client:
            candidates += generate_gemini(sys_p, usr_p, args.n_gemini, gemini_temps)

        # Deduplicate
        seen = set()
        unique_cands = []
        for c in candidates:
            if c["text"] and c["text"] not in seen:
                seen.add(c["text"])
                unique_cands.append(c)

        # Context and ground truth for traceability
        ctx_texts = pair["context_texts"]
        ctx_joined = pair["context_joined"]
        target_text = pair["target_text"]

        # Add candidates as rows
        for ci, c in enumerate(unique_cands):
            all_rows.append({
                "pair_id": pair["pair_id"],
                "context_1": ctx_texts[0] if len(ctx_texts) > 0 else "",
                "context_2": ctx_texts[1] if len(ctx_texts) > 1 else "",
                "context_3": ctx_texts[2] if len(ctx_texts) > 2 else "",
                "context_4": ctx_texts[3] if len(ctx_texts) > 3 else "",
                "context_5": ctx_texts[4] if len(ctx_texts) > 4 else "",
                "ground_truth": target_text,
                "candidate_id": f"{pair['pair_id']}_c{ci:02d}",
                "candidate_text": c["text"],
                "source": c["source"],
                "temperature": c["temp"],
                "is_human": False,
            })

        # Add human (ground truth) response
        all_rows.append({
            "pair_id": pair["pair_id"],
            "context_1": ctx_texts[0] if len(ctx_texts) > 0 else "",
            "context_2": ctx_texts[1] if len(ctx_texts) > 1 else "",
            "context_3": ctx_texts[2] if len(ctx_texts) > 2 else "",
            "context_4": ctx_texts[3] if len(ctx_texts) > 3 else "",
            "context_5": ctx_texts[4] if len(ctx_texts) > 4 else "",
            "ground_truth": target_text,
            "candidate_id": f"{pair['pair_id']}_human",
            "candidate_text": pair["target_text"],
            "source": "human",
            "temperature": 0.0,
            "is_human": True,
        })

        n_s = sum(1 for c in unique_cands if c["source"] == "sarvam")
        n_g = sum(1 for c in unique_cands if c["source"] == "gemini")
        pr(f"    Pair {pi+1}/{len(test_pairs)} ({pair['pair_id']}): {n_s}S + {n_g}G = {len(unique_cands)} unique candidates")

    cand_df = pd.DataFrame(all_rows)
    cand_df.to_csv(CAND_CSV, index=False, encoding="utf-8-sig")
    pr(f"\n  Saved: {CAND_CSV} ({len(cand_df)} rows)")

else:
    # Load existing candidates
    pr(f"\n  Loading existing candidates from {CAND_CSV}...")
    cand_df = pd.read_csv(CAND_CSV, encoding="utf-8-sig")
    pr(f"  Loaded {len(cand_df)} rows")

# ══════════════════════════════════════════════════════════════
#  STEP 6: CCS Scoring + Naturalness Ranking
# ══════════════════════════════════════════════════════════════
pr(f"\n{'─'*65}")
pr("  STEP 6: CCS Scoring + Naturalness Ranking")
pr(f"{'─'*65}")

# Load Stanza if not already loaded (needed for IC/SC of candidates)
if args.prompt_version != 5 or 'nlp_stanza' not in dir():
    pr("\n  Loading Stanza Hindi for CCS scoring...")
    import stanza
    try:
        nlp_stanza = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                                      use_gpu=torch.cuda.is_available(), verbose=False)
    except Exception:
        stanza.download("hi", verbose=False)
        nlp_stanza = stanza.Pipeline("hi", processors="tokenize,pos,lemma,depparse",
                                      use_gpu=torch.cuda.is_available(), verbose=False)
    pr("  Stanza Hindi loaded")

def compute_stanza_ic_sc(text, nlp):
    """Compute mean IC and SC using Stanza parse (same as pipeline)."""
    doc = nlp(text)
    all_ic, all_sc = [], []
    for sent in doc.sentences:
        tokens = []
        for word in sent.words:
            tokens.append({
                "id": word.id,
                "form": word.text,
                "upos": word.upos,
                "head": word.head,
                "deprel": word.deprel,
            })
        ic_list, sc_list = compute_ic_sc_from_tokens(tokens)
        all_ic.extend(ic_list)
        all_sc.extend(sc_list)
    return np.mean(all_ic) if all_ic else 1.0, np.mean(all_sc) if all_sc else 1.0

def compute_naturalness(ccs, ref_m, ref_s):
    """Naturalness = 1 / (1 + |CCS - ref_mean| / ref_sd)"""
    if ref_s == 0:
        return 0.5
    return 1.0 / (1.0 + abs(ccs - ref_m) / ref_s)

pr(f"\n  Scoring {len(cand_df)} candidates...")
ccs_scores = []
nat_scores = []
mean_ics = []
mean_scs = []
fwd_surps = []
bwd_surps = []

for i, row in cand_df.iterrows():
    text = str(row["candidate_text"])
    # IC/SC from Stanza
    m_ic, m_sc = compute_stanza_ic_sc(text, nlp_stanza)
    # Surprisal from GPT-2
    fwd, bwd = compute_surprisal(text, gpt2_tok, gpt2_mdl, device)
    # CCS
    ccs = compute_ccs_score(m_ic, m_sc, fwd, bwd)
    nat = compute_naturalness(ccs, ref_mean, ref_sd)

    ccs_scores.append(ccs)
    nat_scores.append(nat)
    mean_ics.append(m_ic)
    mean_scs.append(m_sc)
    fwd_surps.append(fwd)
    bwd_surps.append(bwd)

    if (i + 1) % 50 == 0:
        pr(f"    Scored {i+1}/{len(cand_df)} | {row['candidate_id']}: CCS={ccs:.6f} Nat={nat:.4f}")

cand_df["mean_ic"] = mean_ics
cand_df["mean_sc"] = mean_scs
cand_df["fwd_surprisal"] = fwd_surps
cand_df["bwd_surprisal"] = bwd_surps
cand_df["ccs_score"] = ccs_scores
cand_df["naturalness"] = nat_scores

# ── Response Selection + MRR ──
pr(f"\n  Response Selection + MRR Computation...")

# Build lookup: pair_id → context and target info
pair_context_lookup = {}
for pair in test_pairs:
    pair_context_lookup[pair["pair_id"]] = pair

pair_ids = cand_df["pair_id"].unique()
selected_rows = []
mrr_scores = []
human_ranks = []

for pi, pid in enumerate(pair_ids):
    pdf = cand_df[cand_df["pair_id"] == pid].copy()
    # Rank by naturalness (descending)
    pdf = pdf.sort_values("naturalness", ascending=False).reset_index(drop=True)
    pdf["rank"] = range(1, len(pdf) + 1)

    # Selected = top-ranked non-human
    llm_only = pdf[~pdf["is_human"]]
    if len(llm_only) > 0:
        sel = llm_only.iloc[0]
    else:
        sel = pdf.iloc[0]

    # Human rank
    human_row = pdf[pdf["is_human"]]
    if len(human_row) > 0:
        h_rank = human_row.iloc[0]["rank"]
        mrr_scores.append(1.0 / h_rank)
        human_ranks.append(h_rank)
    else:
        human_ranks.append(len(pdf))
        mrr_scores.append(0.0)

    # ── Detailed per-pair display ──
    pair_info = pair_context_lookup.get(pid, {})
    pr(f"\n  ┌─ Pair {pi+1}: {pid} ─────────────────────────────")
    if pair_info:
        pr(f"  │ CONTEXT ({args.context} sentences):")
        for ci, ctx in enumerate(pair_info.get("context_texts", [])):
            pr(f"  │   [{ci+1}] {ctx[:100]}")
        pr(f"  │")
        pr(f"  │ GROUND TRUTH (human):")
        pr(f"  │   → {pair_info.get('target_text', '')[:120]}")
        pr(f"  │   CCS={human_row.iloc[0]['ccs_score']:.6f}  Nat={human_row.iloc[0]['naturalness']:.4f}  Rank={h_rank}/{len(pdf)}")
    pr(f"  │")
    pr(f"  │ CCS-SELECTED (rank 1):")
    pr(f"  │   → {sel['candidate_text'][:120]}")
    pr(f"  │   Source={sel['source']}  CCS={sel['ccs_score']:.6f}  Nat={sel['naturalness']:.4f}")
    pr(f"  │")
    pr(f"  │ TOP-3 CANDIDATES:")
    for ri, (_, row) in enumerate(pdf.head(3).iterrows()):
        marker = "★" if row["is_human"] else " "
        pr(f"  │   {ri+1}. [{row['source']:6s}] {marker} {str(row['candidate_text'])[:90]}")
        pr(f"  │      CCS={row['ccs_score']:.6f}  Nat={row['naturalness']:.4f}")
    pr(f"  └────────────────────────────────────────────────────")

    # Store context in selected_rows for CSV
    context_joined = pair_info.get("context_joined", "") if pair_info else ""
    selected_rows.append({
        "pair_id": pid,
        "context": context_joined,
        "selected_id": sel["candidate_id"],
        "selected_text": sel["candidate_text"],
        "selected_ccs": sel["ccs_score"],
        "selected_naturalness": sel["naturalness"],
        "selected_source": sel["source"],
        "human_text": human_row.iloc[0]["candidate_text"] if len(human_row) > 0 else "",
        "human_ccs": human_row.iloc[0]["ccs_score"] if len(human_row) > 0 else 0,
        "human_naturalness": human_row.iloc[0]["naturalness"] if len(human_row) > 0 else 0,
        "human_rank": h_rank if len(human_row) > 0 else len(pdf),
        "n_candidates": len(pdf),
    })

sel_df = pd.DataFrame(selected_rows)
mrr = np.mean(mrr_scores)
avg_rank = np.mean(human_ranks)
top1 = sum(1 for r in human_ranks if r == 1)
top3 = sum(1 for r in human_ranks if r <= 3)

pr(f"\n  MRR                  : {mrr:.4f}")
pr(f"  Avg human rank       : {avg_rank:.1f}")
pr(f"  Human in top-1       : {top1}/{len(pair_ids)} ({100*top1/len(pair_ids):.1f}%)")
pr(f"  Human in top-3       : {top3}/{len(pair_ids)} ({100*top3/len(pair_ids):.1f}%)")
pr(f"  Avg naturalness (sel): {sel_df['selected_naturalness'].mean():.4f}")
pr(f"  Avg naturalness (hum): {sel_df['human_naturalness'].mean():.4f}")

# Save scored candidates
cand_df.to_csv(SCORED_CSV, index=False, encoding="utf-8-sig")
pr(f"\n  Saved: {SCORED_CSV}")

# ══════════════════════════════════════════════════════════════
#  STEP 7: Text-Overlap Metrics (ENHANCED)
# ══════════════════════════════════════════════════════════════
pr(f"\n{'─'*65}")
pr("  STEP 7: Text-Overlap Metrics (ENHANCED)")
pr(f"{'─'*65}")

# Manual ROUGE for Hindi (same approach as Gadkari pipeline)
# The rouge_scorer library's internal tokenizer doesn't handle Hindi/Devanagari well,
# so we compute ROUGE-1 and ROUGE-L manually using space-split tokens and LCS.

def _rouge1_f1(ref_text, hyp_text):
    """Manual ROUGE-1 F1: unigram overlap (works for Hindi space-split tokens)."""
    ref_tokens = ref_text.split()
    hyp_tokens = hyp_text.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    hyp_counts = {}
    for t in hyp_tokens:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1
    overlap = sum(min(ref_counts.get(t, 0), c) for t, c in hyp_counts.items())
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _lcs_length(x, y):
    """Longest Common Subsequence length (for ROUGE-L)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]

def _rougeL_f1(ref_text, hyp_text):
    """Manual ROUGE-L F1: LCS-based (works for Hindi space-split tokens)."""
    ref_tokens = ref_text.split()
    hyp_tokens = hyp_text.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

bleu_scores = []
rouge1_scores = []
rougeL_scores = []
bertscore_f1s = []
bertscore_f1_scaled = []
embedding_cosine_sims = []

# Compute for selected vs human
for _, row in sel_df.iterrows():
    sel_text = str(row["selected_text"])
    hum_text = str(row["human_text"])
    if not hum_text:
        continue

    # BLEU (nltk with space-split — same as Gadkari pipeline)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = hum_text.split()
    hyp_tokens = sel_text.split()
    try:
        bleu = sentence_bleu([ref_tokens], hyp_tokens,
                              smoothing_function=SmoothingFunction().method1)
    except Exception:
        bleu = 0.0
    bleu_scores.append(bleu)

    # ROUGE (manual functions — handles Hindi properly)
    rouge1_scores.append(_rouge1_f1(hum_text, sel_text))
    rougeL_scores.append(_rougeL_f1(hum_text, sel_text))

# BERTScore for all candidates vs human
pr(f"\n  Computing BERTScore F1 for all candidates...")
all_cand_texts = cand_df["candidate_text"].tolist()
# Build reference list: human text for each candidate's pair
all_refs = []
human_by_pair = {}
for _, row in cand_df[cand_df["is_human"]].iterrows():
    human_by_pair[row["pair_id"]] = str(row["candidate_text"])
for _, row in cand_df.iterrows():
    all_refs.append(human_by_pair.get(row["pair_id"], ""))

all_bertscore_raw = []
all_bertscore_scaled = []
all_emb_cosine = []

try:
    from bert_score import score as bert_score_fn
    _, _, F1 = bert_score_fn(all_cand_texts, all_refs, lang="hi", verbose=False)
    all_bertscore_raw = F1.tolist()
    pr(f"    Raw BERTScore F1 avg: {np.mean(all_bertscore_raw):.4f}")
except Exception as e:
    pr(f"  [WARN] BERTScore failed: {e}")

# Scaled BERTScore — HDTB-specific baseline
#
# WHY NOT USE GADKARI BASELINE (0.6499)?
#   Gadkari baseline was computed on conversational Hindi speech (38 pairs).
#   HDTB is written news text — different domain, vocabulary, sentence structure.
#   Using Gadkari's baseline would underestimate the true baseline for news text,
#   making scaled scores artificially high. Domain-specific baseline is more accurate.
#
# HOW HDTB BASELINE IS COMPUTED:
#   Take 500 random sentence pairs from HDTB train (non-adjacent, seed=42).
#   Compute BERTScore F1 for each random pair using bert-base-multilingual-cased.
#   Average = expected BERTScore between unrelated Hindi news sentences.
#   This is a one-time computation; once known, hardcode it above as
#   PRECOMPUTED_BERTSCORE_BASELINE_HDTB.
#
if all_bertscore_raw:
    if PRECOMPUTED_BERTSCORE_BASELINE_HDTB is not None:
        baseline = PRECOMPUTED_BERTSCORE_BASELINE_HDTB
        pr(f"  Using PRECOMPUTED HDTB BERTScore baseline: {baseline:.4f}")
        pr(f"    (from HDTB train, 500 random pairs, bert-base-multilingual-cased)")
    else:
        # Compute HDTB-specific baseline from train sentences
        pr(f"  Computing HDTB-specific BERTScore baseline from train...")
        pr(f"    (500 random non-adjacent sentence pairs, seed=42)")
        baseline_rng = np.random.RandomState(42)
        n_baseline_pairs = 500
        train_texts = [s["text"] for s in train_sents]
        idx_a = baseline_rng.choice(len(train_texts), n_baseline_pairs, replace=False)
        idx_b = baseline_rng.choice(len(train_texts), n_baseline_pairs, replace=False)
        # Ensure pairs are non-adjacent (at least 10 sentences apart)
        for k in range(n_baseline_pairs):
            while abs(idx_a[k] - idx_b[k]) < 10:
                idx_b[k] = baseline_rng.randint(0, len(train_texts))
        baseline_texts_a = [train_texts[i] for i in idx_a]
        baseline_texts_b = [train_texts[i] for i in idx_b]
        try:
            _, _, baseline_F1 = bert_score_fn(baseline_texts_a, baseline_texts_b,
                                               lang="hi", verbose=False)
            baseline = float(baseline_F1.mean())
            pr(f"    HDTB BERTScore baseline: {baseline:.4f}")
            pr(f"    *** UPDATE PRECOMPUTED_BERTSCORE_BASELINE_HDTB = {baseline:.4f} in code ***")
        except Exception as e:
            pr(f"    [WARN] Baseline computation failed: {e}")
            pr(f"    Falling back to Gadkari baseline: 0.6499")
            baseline = 0.6499

    all_bertscore_scaled = [
        (raw - baseline) / (1 - baseline) if (1 - baseline) > 0 else 0.0
        for raw in all_bertscore_raw
    ]
    pr(f"    Scaled BERTScore F1 avg: {np.mean(all_bertscore_scaled):.4f}")

# Add to scored_df
if all_bertscore_raw and len(all_bertscore_raw) == len(cand_df):
    cand_df["bertscore_f1_raw"] = all_bertscore_raw
if all_bertscore_scaled and len(all_bertscore_scaled) == len(cand_df):
    cand_df["bertscore_f1_scaled"] = all_bertscore_scaled

# Embedding Cosine Similarity
pr(f"\n  ── Embedding Cosine Similarity ──")
try:
    from transformers import AutoModel
    emb_candidates = ["xlm-roberta-base", "bert-base-multilingual-cased"]
    emb_device = device
    emb_model = None
    emb_tokenizer = None
    emb_model_name = None

    for mname in emb_candidates:
        try:
            emb_tokenizer = AutoTokenizer.from_pretrained(mname)
            emb_model = AutoModel.from_pretrained(mname)
            emb_model.eval().to(emb_device)
            emb_model_name = mname
            pr(f"  Embedding model loaded: {mname}")
            break
        except Exception as e:
            pr(f"    ✗ {mname}: {str(e)[:60]}")

    if emb_model:
        def get_embedding(text):
            enc = emb_tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=512, padding=True).to(emb_device)
            with torch.no_grad():
                out = emb_model(**enc)
            # Mean pooling
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            return emb.squeeze().cpu().numpy()

        # Compute for all candidates
        pr(f"  Computing embedding similarities for all {len(cand_df)} candidates...")
        human_embs = {}
        for pid, htxt in human_by_pair.items():
            human_embs[pid] = get_embedding(htxt)

        for i, row in cand_df.iterrows():
            pid = row["pair_id"]
            if pid in human_embs:
                c_emb = get_embedding(str(row["candidate_text"]))
                h_emb = human_embs[pid]
                cos_sim = float(np.dot(c_emb, h_emb) / (np.linalg.norm(c_emb) * np.linalg.norm(h_emb) + 1e-9))
                all_emb_cosine.append(cos_sim)
            else:
                all_emb_cosine.append(0.0)

        pr(f"    Avg embedding cosine similarity: {np.mean(all_emb_cosine):.4f}")

        if len(all_emb_cosine) == len(cand_df):
            cand_df["embedding_cosine_sim"] = all_emb_cosine

except Exception as e:
    pr(f"  [WARN] Embedding similarity failed: {e}")

# Compute BLEU/ROUGE for each candidate vs ground truth
pr(f"\n  Computing BLEU/ROUGE for all candidates...")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
cand_bleu = []
cand_rouge1 = []
cand_rougeL = []
for _, row in cand_df.iterrows():
    cand_text = str(row["candidate_text"])
    hum_text = str(row.get("ground_truth", ""))
    if not hum_text or not cand_text:
        cand_bleu.append(0.0)
        cand_rouge1.append(0.0)
        cand_rougeL.append(0.0)
        continue
    ref_tokens = hum_text.split()
    hyp_tokens = cand_text.split()
    try:
        b = sentence_bleu([ref_tokens], hyp_tokens,
                          smoothing_function=SmoothingFunction().method1)
    except Exception:
        b = 0.0
    cand_bleu.append(b)
    cand_rouge1.append(_rouge1_f1(hum_text, cand_text))
    cand_rougeL.append(_rougeL_f1(hum_text, cand_text))
cand_df["bleu"] = cand_bleu
cand_df["rouge1"] = cand_rouge1
cand_df["rougeL"] = cand_rougeL

# Re-save with all metrics
cand_df.to_csv(SCORED_CSV, index=False, encoding="utf-8-sig")
pr(f"\n  Updated: {SCORED_CSV}")

# Get ALL metrics for selected responses from cand_df
sel_bertscore_raw = []
sel_bertscore_scaled = []
sel_emb_cosine = []
sel_bleu = []
sel_rouge1 = []
sel_rougeL = []
for _, row in sel_df.iterrows():
    sel_id = row["selected_id"]
    sel_match = cand_df[cand_df["candidate_id"] == sel_id]
    if len(sel_match) > 0:
        m = sel_match.iloc[0]
        sel_bertscore_raw.append(m.get("bertscore_f1_raw", 0.0))
        sel_bertscore_scaled.append(m.get("bertscore_f1_scaled", 0.0))
        sel_emb_cosine.append(m.get("embedding_cosine_sim", 0.0))
        sel_bleu.append(m.get("bleu", 0.0))
        sel_rouge1.append(m.get("rouge1", 0.0))
        sel_rougeL.append(m.get("rougeL", 0.0))
    else:
        sel_bertscore_raw.append(0.0)
        sel_bertscore_scaled.append(0.0)
        sel_emb_cosine.append(0.0)
        sel_bleu.append(0.0)
        sel_rouge1.append(0.0)
        sel_rougeL.append(0.0)

# Also use the directly computed BLEU/ROUGE (selected vs human) if available
# These are more accurate as they're computed directly, not from cand_df
if bleu_scores and len(bleu_scores) == len(sel_df):
    sel_bleu = bleu_scores
if rouge1_scores and len(rouge1_scores) == len(sel_df):
    sel_rouge1 = rouge1_scores
    sel_rougeL = rougeL_scores

# Add all metric columns to selected_responses
sel_df["bleu"] = sel_bleu
sel_df["rouge1"] = sel_rouge1
sel_df["rougeL"] = sel_rougeL
sel_df["bertscore_f1"] = sel_bertscore_raw
sel_df["bertscore_f1_scaled"] = sel_bertscore_scaled
sel_df["embedding_cosine_sim"] = sel_emb_cosine

# Update lists for final report
bertscore_f1s = sel_bertscore_raw
bertscore_f1_scaled = sel_bertscore_scaled
embedding_cosine_sims = sel_emb_cosine

# Save selected responses
sel_df.to_csv(SELECT_CSV, index=False, encoding="utf-8-sig")
pr(f"  Saved: {SELECT_CSV}")

# ── Save detailed per-pair analysis report ──
DETAIL_TXT = os.path.join(OUT_DIR, "pair_detail_report.txt")
with open(DETAIL_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("  HDTB PIPELINE — DETAILED PER-PAIR ANALYSIS\n")
    f.write(f"  V{args.prompt_version} Prompt | Context={args.context} sentences | {len(pair_ids)} pairs\n")
    f.write(f"  CCS Reference: mean={ref_mean:.6f}, sd={ref_sd:.6f}\n")
    f.write("=" * 80 + "\n\n")

    for pi, pid in enumerate(pair_ids):
        pair_info = pair_context_lookup.get(pid, {})
        pdf = cand_df[cand_df["pair_id"] == pid].copy()
        pdf = pdf.sort_values("naturalness", ascending=False).reset_index(drop=True)
        pdf["rank"] = range(1, len(pdf) + 1)

        human_row = pdf[pdf["is_human"]]
        h_rank = int(human_row.iloc[0]["rank"]) if len(human_row) > 0 else len(pdf)

        f.write(f"{'─'*80}\n")
        f.write(f"  PAIR {pi+1}/{len(pair_ids)}: {pid}\n")
        f.write(f"{'─'*80}\n\n")

        # Context
        if pair_info:
            f.write(f"  INPUT CONTEXT ({args.context} sentences given to LLM):\n")
            for ci, ctx in enumerate(pair_info.get("context_texts", [])):
                f.write(f"    [{ci+1}] {ctx}\n")
            f.write(f"\n")

            # Ground truth
            f.write(f"  GROUND TRUTH (actual next sentence from HDTB corpus):\n")
            f.write(f"    → {pair_info.get('target_text', '')}\n")
            if len(human_row) > 0:
                hr = human_row.iloc[0]
                f.write(f"    CCS Score    : {hr['ccs_score']:.6f}\n")
                f.write(f"    Naturalness  : {hr['naturalness']:.4f}\n")
                f.write(f"    Rank         : {h_rank}/{len(pdf)} (among all candidates + human)\n")
                if "bertscore_f1_raw" in hr.index and pd.notna(hr.get("bertscore_f1_raw")):
                    f.write(f"    BERTScore F1 : {hr['bertscore_f1_raw']:.4f} (raw) | {hr.get('bertscore_f1_scaled', 0):.4f} (scaled)\n")
            f.write(f"\n")

        # All candidates ranked
        f.write(f"  ALL CANDIDATES (ranked by CCS naturalness, highest = most natural):\n")
        f.write(f"  {'Rank':<5} {'Source':<8} {'Nat':>6} {'CCS':>10} {'BERTsc':>8} {'EmbCos':>8}  Text\n")
        f.write(f"  {'─'*5} {'─'*8} {'─'*6} {'─'*10} {'─'*8} {'─'*8}  {'─'*40}\n")

        for ri, (_, row) in enumerate(pdf.iterrows()):
            marker = " ★ HUMAN" if row["is_human"] else ""
            sel_marker = " ◆ SELECTED" if not row["is_human"] and ri == 0 else ""
            bert_raw = f"{row['bertscore_f1_raw']:.4f}" if "bertscore_f1_raw" in row.index and pd.notna(row.get("bertscore_f1_raw")) else "  —   "
            emb_cos = f"{row['embedding_cosine_sim']:.4f}" if "embedding_cosine_sim" in row.index and pd.notna(row.get("embedding_cosine_sim")) else "  —   "
            f.write(f"  {ri+1:<5} {row['source']:<8} {row['naturalness']:>6.4f} {row['ccs_score']:>10.6f} {bert_raw:>8} {emb_cos:>8}  {str(row['candidate_text'])[:100]}{marker}{sel_marker}\n")

        f.write(f"\n")
        f.write(f"  SUMMARY: Human ranked {h_rank}/{len(pdf)} | ")
        f.write(f"Selected source: {pdf.iloc[0]['source'] if not pdf.iloc[0]['is_human'] else 'human'}\n")
        f.write(f"\n\n")

pr(f"  Saved: {DETAIL_TXT} (detailed per-pair analysis)")

# ══════════════════════════════════════════════════════════════
#  STEP 8 (Optional): Gold vs Stanza IC/SC Comparison
# ══════════════════════════════════════════════════════════════
if args.gold_vs_stanza:
    pr(f"\n{'─'*65}")
    pr("  STEP 8: Gold vs Stanza IC/SC Comparison")
    pr(f"{'─'*65}")

    # Sample test sentences for comparison
    n_compare = min(200, len(test_sents))
    compare_sents = test_sents[:n_compare]

    gold_ics, gold_scs = [], []
    stanza_ics, stanza_scs = [], []

    for si, sent in enumerate(compare_sents):
        # Gold IC/SC
        g_feats = compute_sentence_ccs_features(sent["tokens"])
        gold_ics.append(g_feats["mean_ic"])
        gold_scs.append(g_feats["mean_sc"])

        # Stanza IC/SC
        s_ic, s_sc = compute_stanza_ic_sc(sent["text"], nlp_stanza)
        stanza_ics.append(s_ic)
        stanza_scs.append(s_sc)

        if (si + 1) % 50 == 0:
            pr(f"    Compared {si+1}/{n_compare} sentences...")

    # Correlation
    from scipy import stats
    ic_corr, ic_p = stats.pearsonr(gold_ics, stanza_ics)
    sc_corr, sc_p = stats.pearsonr(gold_scs, stanza_scs)
    ic_mae = np.mean(np.abs(np.array(gold_ics) - np.array(stanza_ics)))
    sc_mae = np.mean(np.abs(np.array(gold_scs) - np.array(stanza_scs)))

    pr(f"\n  Gold vs Stanza IC/SC Comparison ({n_compare} sentences):")
    pr(f"    IC: Pearson r = {ic_corr:.4f} (p = {ic_p:.2e}), MAE = {ic_mae:.4f}")
    pr(f"    SC: Pearson r = {sc_corr:.4f} (p = {sc_p:.2e}), MAE = {sc_mae:.4f}")
    pr(f"    Gold IC  mean: {np.mean(gold_ics):.4f}, Stanza IC  mean: {np.mean(stanza_ics):.4f}")
    pr(f"    Gold SC  mean: {np.mean(gold_scs):.4f}, Stanza SC  mean: {np.mean(stanza_scs):.4f}")

# ══════════════════════════════════════════════════════════════
#  STEP 9: TTS (Text-to-Speech) — generate audio for selected responses
#  Uses Facebook MMS-TTS Hindi (facebook/mms-tts-hin), same as Gadkari pipeline.
#  Generates .wav files for each selected response so Prof can listen.
# ══════════════════════════════════════════════════════════════
pr(f"\n{'─'*65}")
pr("  STEP 9: TTS (Text-to-Speech)")
pr(f"{'─'*65}")

audio_dir = os.path.join(OUT_DIR, "audio")
os.makedirs(audio_dir, exist_ok=True)
try:
    from transformers import VitsModel, AutoTokenizer as VitsTokenizer
    import scipy.io.wavfile as wavfile
    tts_model_name = "facebook/mms-tts-hin"
    pr(f"\n  Loading TTS model: {tts_model_name}...")
    tts_model = VitsModel.from_pretrained(tts_model_name)
    tts_tokenizer = VitsTokenizer.from_pretrained(tts_model_name)
    tts_model.eval()
    pr("  TTS model loaded")
    tts_count = 0
    for _, row in sel_df.iterrows():
        pair_id = row["pair_id"]
        text = str(row["selected_text"]).strip()
        if not text:
            continue
        inputs = tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs)
        waveform = output.waveform[0].cpu().numpy()
        sample_rate = tts_model.config.sampling_rate
        safe_name = pair_id.replace(" ", "_").replace("/", "_")
        wav_path = os.path.join(audio_dir, f"{safe_name}.wav")
        wavfile.write(wav_path, rate=sample_rate, data=waveform)
        tts_count += 1
    pr(f"\n  TTS: {tts_count} audio files generated in {audio_dir}")
except ImportError as e:
    pr(f"\n  [WARN] TTS skipped — missing package: {e}")
    audio_dir = None
except Exception as e:
    pr(f"\n  [WARN] TTS failed: {e}")
    audio_dir = None

# ══════════════════════════════════════════════════════════════
#  FINAL REPORT
# ══════════════════════════════════════════════════════════════
elapsed = time.time() - t0
pr(f"\n{'='*65}")
pr(f"  HDTB EVALUATION RESULTS (V{args.prompt_version}, ctx={args.context})")
pr(f"{'='*65}")
pr(f"\n  Data source            : UD Hindi-HDTB (news/written text)")
pr(f"  Total test pairs       : {len(test_pairs)}")
pr(f"  Prompt version         : V{args.prompt_version}")
pr(f"  Context window         : {args.context} sentences")
pr(f"  CCS mode               : Linguistic-only (IC, SC, fwd/bwd surprisal)")
pr(f"  CCS Reference (train)  : mean={ref_mean:.6f}, sd={ref_sd:.6f}")
pr(f"\n  Selection Results:")
pr(f"    Pairs evaluated      : {len(sel_df)}")
pr(f"    Avg naturalness (sel): {sel_df['selected_naturalness'].mean():.4f}")
pr(f"    Avg naturalness (hum): {sel_df['human_naturalness'].mean():.4f}")
pr(f"\n  *** MRR (Mean Reciprocal Rank) ***")
pr(f"    MRR                  : {mrr:.4f}")
pr(f"    Avg human rank       : {avg_rank:.1f}")
pr(f"    Human in top-1       : {top1}/{len(pair_ids)} ({100*top1/len(pair_ids):.1f}%)")
pr(f"    Human in top-3       : {top3}/{len(pair_ids)} ({100*top3/len(pair_ids):.1f}%)")
pr(f"\n  *** Text-Overlap Metrics (selected vs human) ***")
if bleu_scores:
    pr(f"    Avg BLEU                    : {np.mean(bleu_scores):.4f}")
if rouge1_scores:
    pr(f"    Avg ROUGE-1 (F1)            : {np.mean(rouge1_scores):.4f}")
    pr(f"    Avg ROUGE-L (F1)            : {np.mean(rougeL_scores):.4f}")
if bertscore_f1s:
    pr(f"    Avg BERTScore F1 (raw)      : {np.mean(bertscore_f1s):.4f}")
if bertscore_f1_scaled:
    pr(f"    Avg BERTScore F1 (SCALED)   : {np.mean(bertscore_f1_scaled):.4f}")
if embedding_cosine_sims:
    pr(f"    Avg Embedding Cosine Sim    : {np.mean(embedding_cosine_sims):.4f}")

if all_bertscore_raw:
    pr(f"\n  All Candidates vs Human:")
    pr(f"    Avg BERTScore F1 (raw)      : {np.mean(all_bertscore_raw):.4f}")
if all_bertscore_scaled:
    pr(f"    Avg BERTScore F1 (SCALED)   : {np.mean(all_bertscore_scaled):.4f}")
if all_emb_cosine:
    pr(f"    Avg Embedding Cosine Sim    : {np.mean(all_emb_cosine):.4f}")

pr(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
pr(f"\n  Outputs:")
pr(f"    {CAND_CSV}")
pr(f"    {SCORED_CSV}")
pr(f"    {SELECT_CSV}")
pr(f"    {DETAIL_TXT}")
pr(f"    {REPORT_TXT}")
pr(f"{'='*65}")

# Save report
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(_log_lines))
pr(f"\n  Report saved: {REPORT_TXT}")