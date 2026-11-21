#!/usr/bin/env python3
"""
phase_C_surprisal.py — Phase C: Surprisal Features (GPT-2 Hindi)

Uses MTP-2 approach (extract_surprisal_v4.py logic):
  ONE forward pass per segment (not per word) — much faster
  Multi-token words: surprisal = sum of token surprisals
  Chain rule of probability (Ranjan SCiL 2022)

Computes:
  forward_surprisal  = -log2 P(word | preceding words)
  backward_surprisal = -log2 P(word | following words)

INPUT : outputs/results/word_features_v4.csv (614,351 × 95)
        outputs/results/indicvoices_v1_hindi_segments_v4.csv
OUTPUT: outputs/results/features_C_surprisal.csv (614,351 × 4)
        columns: segment_id, word_index,
                 forward_surprisal, backward_surprisal

Time  : ~2-3 hours on GPU (28,302 segments)
Resume: python phase_C_surprisal.py --resume_seg N

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import csv
import math
import time
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT       = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(ROOT, "outputs", "results")
WORD_CSV   = os.path.join(RESULTS, "word_features_v4.csv")
SEG_CSV    = os.path.join(RESULTS,
             "indicvoices_v1_hindi_segments_v4.csv")
OUTPUT_CSV = os.path.join(RESULTS, "features_C_surprisal.csv")
GPT2_MODEL = "surajp/gpt2-hindi"

FIELDNAMES = [
    "segment_id", "word_index",
    "forward_surprisal", "backward_surprisal",
]


def clean_text_for_gpt2(text):
    """Basic cleaning — remove extra spaces."""
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_token_logprobs(tokenizer, model, word_list, device):
    """
    ONE forward pass for entire word sequence.
    Returns per-token (token_id, surprisal_bits) list.
    Uses shift trick: logits[i] predicts token[i+1].
    """
    import torch
    text = " ".join(word_list)
    enc  = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    if input_ids.shape[1] < 2:
        return []

    with torch.no_grad():
        outputs  = model(input_ids)
        logits   = outputs.logits  # (1, seq_len, vocab_size)

    log_probs  = torch.nn.functional.log_softmax(
        logits[0, :-1, :], dim=-1)
    target_ids = input_ids[0, 1:]

    token_logprobs = []
    for i, tid in enumerate(target_ids):
        lp_nats = log_probs[i, tid].item()
        lp_bits = lp_nats / math.log(2)
        token_logprobs.append((tid.item(), -lp_bits))

    return token_logprobs


def map_tokens_to_words(tokenizer, word_list, token_surps):
    """
    Align per-token surprisals back to words.
    Multi-token words: sum all token surprisals (chain rule).
    """
    word_surps = []
    tok_idx    = 0

    for word in word_list:
        word_enc = tokenizer(
            " " + word,
            add_special_tokens=False
        )["input_ids"]
        n_toks = len(word_enc)

        if tok_idx + n_toks <= len(token_surps):
            word_s = sum(
                s for _, s in token_surps[tok_idx:tok_idx+n_toks])
            word_surps.append(round(word_s, 4))
        else:
            word_surps.append(0.0)

        tok_idx += n_toks

    return word_surps


def compute_surprisal(tokenizer, model, words, device):
    """
    Compute backward and forward surprisal for a word list.
    ONE forward pass for backward, ONE for forward (reversed).
    Returns list of (backward_surprisal, forward_surprisal).
    """
    if not words:
        return [(0.0, 0.0)] * len(words)

    # Backward surprisal (normal sentence order)
    try:
        bwd_token_surps = get_token_logprobs(
            tokenizer, model, words, device)
        bwd_surps = map_tokens_to_words(
            tokenizer, words, bwd_token_surps)
    except:
        bwd_surps = [0.0] * len(words)

    # Forward surprisal (reversed sentence)
    try:
        rev_words       = words[::-1]
        fwd_token_surps = get_token_logprobs(
            tokenizer, model, rev_words, device)
        fwd_surps_rev   = map_tokens_to_words(
            tokenizer, rev_words, fwd_token_surps)
        fwd_surps       = fwd_surps_rev[::-1]
    except:
        fwd_surps = [0.0] * len(words)

    return list(zip(bwd_surps, fwd_surps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_seg", type=int, default=0,
        help="Resume from segment number (0=start)")
    args = parser.parse_args()

    print("="*60)
    print("  PHASE C: Surprisal Features (GPT-2 Hindi)")
    print("  Method: MTP-2 approach — 1 pass per segment")
    print("  forward_surprisal, backward_surprisal")
    print("="*60)

    start = time.time()

    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("ERROR: transformers not installed")
        return

    # Load segment text
    print(f"\n  Loading segment metadata ...")
    seg_text = {}
    with open(SEG_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seg_text[row["segment_id"]] = \
                row.get("clean_hindi_text", "").strip()
    print(f"  Segments loaded: {len(seg_text):,}")

    # Load word rows grouped by segment
    print(f"\n  Loading word_features_v4.csv ...")
    seg_words  = defaultdict(list)
    seg_order  = []
    total_words = 0
    with open(WORD_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row["segment_id"]
            if sid not in seg_words:
                seg_order.append(sid)
            seg_words[sid].append(row["word_index"])
            total_words += 1
    print(f"  Words loaded    : {total_words:,}")
    print(f"  Segments        : {len(seg_order):,}")

    # Load GPT-2
    print(f"\n  Loading GPT-2 Hindi ({GPT2_MODEL}) ...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
    model     = GPT2LMHeadModel.from_pretrained(GPT2_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  GPT-2 loaded on {device} ✅")

    # Resume support
    resume_from = args.resume_seg
    mode = "a" if resume_from > 0 else "w"
    if resume_from > 0:
        print(f"\n  Resuming from segment {resume_from:,}")

    total_segs  = len(seg_order)
    done_segs   = 0
    done_words  = 0
    errors      = 0

    with open(OUTPUT_CSV, mode, encoding="utf-8-sig",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if resume_from == 0:
            writer.writeheader()

        for si, seg_id in enumerate(seg_order):
            if si < resume_from:
                done_segs += 1
                continue

            word_indices = seg_words[seg_id]
            text = seg_text.get(seg_id, "").strip()

            if not text:
                for wi in word_indices:
                    writer.writerow({
                        "segment_id"        : seg_id,
                        "word_index"        : wi,
                        "forward_surprisal" : "",
                        "backward_surprisal": "",
                    })
                done_words += len(word_indices)
                done_segs  += 1
                continue

            words = clean_text_for_gpt2(text).split()

            try:
                surp_pairs = compute_surprisal(
                    tokenizer, model, words, device)
            except Exception as e:
                surp_pairs = [(0.0, 0.0)] * len(words)
                errors += 1

            for pos, wi in enumerate(word_indices):
                if pos < len(surp_pairs):
                    bwd, fwd = surp_pairs[pos]
                else:
                    bwd, fwd = 0.0, 0.0

                writer.writerow({
                    "segment_id"        : seg_id,
                    "word_index"        : wi,
                    "forward_surprisal" : fwd,
                    "backward_surprisal": bwd,
                })

            done_words += len(word_indices)
            done_segs  += 1

            if done_segs % 100 == 0:
                pct     = round(100*done_segs/total_segs, 1)
                elapsed = round(time.time()-start, 0)
                rate    = done_segs / max(elapsed, 1)
                eta     = round(
                    (total_segs-done_segs)/max(rate,0.01), 0)
                print(f"  [{done_segs:,}/{total_segs:,}] {pct}%"
                      f" | elapsed={int(elapsed)}s"
                      f" ETA={int(eta)}s"
                      f" | errors={errors}"
                      f" | resume: --resume_seg {done_segs}")
                f.flush()

    sz      = round(os.path.getsize(OUTPUT_CSV)/1024/1024, 1)
    elapsed = round(time.time()-start, 1)

    print()
    print("="*60)
    print("  PHASE C COMPLETE")
    print("="*60)
    print(f"  Words processed : {done_words:,}")
    print(f"  Segs processed  : {done_segs:,}")
    print(f"  Errors          : {errors:,}")
    print(f"  Columns         : {len(FIELDNAMES)}")
    print(f"  Size            : {sz} MB")
    print(f"  Time            : {elapsed}s ({round(elapsed/3600,1)}h)")
    print(f"  Output          : {OUTPUT_CSV}")
    print("="*60)


if __name__ == "__main__":
    main()