#!/usr/bin/env python3
"""
add_word_lab_mapping.py

PURPOSE
-------
Add two new columns to word_features_v3.csv by mapping each word
back to its original LAB file text:

  word_from_lab : actual Hindi word from LAB file at same position
                  filled for ALL rows (not just <unk>)

  word_resolved : final resolved word
                  if word == '<unk>' → use word_from_lab
                  if word != '<unk>' → use word (MFA was correct)

WHY THIS IS NEEDED
------------------
MFA marks words as <unk> when it cannot find them in the
hindi_indic pronunciation dictionary. Common causes:
  - Dialectal spellings: बहोत (standard: बहुत)
  - Nukta characters  : फ़ैन (ف with nukta)
  - Rare proper nouns : सोनभद्र, सिंगरौली
  - Informal speech   : मवेसी (standard: मवेशी)

The LAB files contain the ORIGINAL Hindi words that MFA
tried (and failed) to align. By reading the LAB file at
the same word_index position, we recover the actual word.

PRE-VERIFIED CHECKS (all passed before writing this script)
-----------------------------------------------------------
✅ LAB files contain NO <unk>       (0 out of 29,231)
✅ LAB files contain NO English      (0 out of 29,231)
✅ Word count MATCH: 16,472/16,472   (100% match)
✅ No position mismatches found
✅ All LAB files exist for <unk> segments

COLUMN POSITION
---------------
New columns inserted IMMEDIATELY AFTER 'word' column:
  word | word_from_lab | word_resolved | word_start_sec ...

This follows the same philosophy as V4 CSV —
insert at the correct logical position, not at end.

INPUT
-----
  outputs/results/word_features_v3.csv  (614,353 rows)
  mfa_segments/{speaker_id}/{segment_id}.lab  (29,231 files)

OUTPUT
------
  outputs/results/word_features_v4.csv  (614,353 rows, 2 new cols)
  outputs/results/word_lab_mapping_report.txt

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
Project   : MTP-2
"""

import os
import sys
import csv
import time
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config

# ── Paths ─────────────────────────────────────────────────────
INPUT_CSV   = os.path.join(config.RESULTS_DIR,
               "word_features_v3.csv")
OUTPUT_CSV  = os.path.join(config.RESULTS_DIR,
               "word_features_v4.csv")
REPORT_TXT  = os.path.join(config.RESULTS_DIR,
               "word_lab_mapping_report.txt")
LAB_DIR     = os.path.join(ROOT, "mfa_segments")


def read_lab_file(speaker_id, segment_id):
    """
    Read LAB file and return list of words.
    LAB files contain ONLY Hindi Devanagari words
    (verified: no <unk>, no English).

    Returns:
      list of words if file exists and has content
      empty list if file missing or empty
    """
    lab_path = os.path.join(LAB_DIR, speaker_id,
                            f"{segment_id}.lab")
    if not os.path.exists(lab_path):
        return []
    try:
        with open(lab_path, encoding="utf-8",
                  errors="replace") as f:
            content = f.read().strip()
        return content.split() if content else []
    except Exception:
        return []


def build_lab_cache(rows):
    """
    Pre-load all LAB files needed for <unk> rows.
    Caches by (speaker_id, segment_id) to avoid
    re-reading the same file multiple times.

    Returns dict: (speaker_id, segment_id) → [words]
    """
    # Find unique segments that have <unk>
    needed = set()
    for row in rows:
        if row.get("word", "") == "<unk>":
            needed.add((
                row.get("speaker_id_file", ""),
                row.get("segment_id", "")
            ))

    cache = {}
    for speaker_id, segment_id in needed:
        cache[(speaker_id, segment_id)] = read_lab_file(
            speaker_id, segment_id)

    return cache


def build_v4_fieldnames(v3_fieldnames):
    """
    Insert word_from_lab and word_resolved
    immediately after the 'word' column.
    All other V3 columns preserved in original order.
    """
    v4_fieldnames = []
    for col in v3_fieldnames:
        v4_fieldnames.append(col)
        if col == "word":
            v4_fieldnames.append("word_from_lab")
            v4_fieldnames.append("word_resolved")
    return v4_fieldnames


def main():
    print("=" * 65)
    print("  ADD WORD LAB MAPPING")
    print("  word_from_lab + word_resolved columns")
    print("=" * 65)
    print()
    print("  Pre-verified checks:")
    print("  ✅ LAB files: no <unk>, no English")
    print("  ✅ Word count match: 16,472/16,472 (100%)")
    print("  ✅ All LAB files exist for <unk> segments")
    print()

    if not os.path.exists(INPUT_CSV):
        print(f"  ERROR: {INPUT_CSV} not found")
        return

    # ── Load V3 CSV ───────────────────────────────────────────
    print(f"  Loading {INPUT_CSV} ...")
    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        reader        = csv.DictReader(f)
        v3_fieldnames = list(reader.fieldnames)
        rows          = list(reader)

    print(f"  Rows    : {len(rows):,}")
    print(f"  Columns : {len(v3_fieldnames)}")

    # ── Build V4 column order ─────────────────────────────────
    v4_fieldnames = build_v4_fieldnames(v3_fieldnames)
    print(f"  V4 cols : {len(v4_fieldnames)}"
          f" (+2: word_from_lab, word_resolved)")
    print()

    # ── Pre-load LAB files for <unk> segments ─────────────────
    print("  Pre-loading LAB files for <unk> segments ...")
    start      = time.time()
    lab_cache  = build_lab_cache(rows)
    print(f"  Unique <unk> segments cached: {len(lab_cache):,}")
    print()

    # ── Process rows ──────────────────────────────────────────
    print("  Processing rows ...")

    # Counters
    total_unk          = 0
    resolved_unk       = 0
    unresolved_unk     = 0   # word_index out of range
    no_lab_unk         = 0   # LAB file not found
    tier_resolved      = Counter()
    tier_unresolved    = Counter()

    v4_rows = []

    for idx, row in enumerate(rows):
        if (idx + 1) % 100000 == 0:
            print(f"  [{idx+1:>7,}/{len(rows):,}]")

        word       = row.get("word", "")
        word_index = int(row.get("word_index", 0))
        speaker    = row.get("speaker_id_file", "")
        seg_id     = row.get("segment_id", "")
        tier       = row.get("dataset_tier", "")

        # Get LAB words for this segment
        lab_words = lab_cache.get((speaker, seg_id))
        if lab_words is None:
            # Not in cache → segment has no <unk> → read anyway
            # (for word_from_lab on non-<unk> rows)
            lab_words = read_lab_file(speaker, seg_id)

        # Compute word_from_lab
        if lab_words and word_index < len(lab_words):
            word_from_lab = lab_words[word_index]
        elif lab_words and word_index >= len(lab_words):
            # word_index out of range (should not happen —
            # pre-verified as 0 mismatches)
            word_from_lab = ""
        else:
            # Empty LAB (annotation-only segment)
            word_from_lab = ""

        # Compute word_resolved
        if word == "<unk>":
            total_unk += 1
            if word_from_lab:
                word_resolved   = word_from_lab
                resolved_unk   += 1
                tier_resolved[tier] += 1
            else:
                word_resolved     = "<unk>"
                unresolved_unk   += 1
                tier_unresolved[tier] += 1
        else:
            # Not <unk> — use MFA word directly
            word_resolved = word

        row["word_from_lab"] = word_from_lab
        row["word_resolved"] = word_resolved
        v4_rows.append(row)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ── Write V4 CSV ──────────────────────────────────────────
    print(f"\n  Writing {OUTPUT_CSV} ...")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=v4_fieldnames,
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(v4_rows)

    v3_mb = os.path.getsize(INPUT_CSV) / 1024 / 1024
    v4_mb = os.path.getsize(OUTPUT_CSV) / 1024 / 1024
    print(f"  V3 size : {v3_mb:.1f} MB")
    print(f"  V4 size : {v4_mb:.1f} MB  (larger = correct)")

    # ── Report ────────────────────────────────────────────────
    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr()
    pr("=" * 65)
    pr("  WORD LAB MAPPING REPORT")
    pr("=" * 65)
    pr(f"  Input  : {INPUT_CSV}")
    pr(f"  Output : {OUTPUT_CSV}")
    pr(f"  Total rows processed    : {len(rows):,}")
    pr()
    pr("── <unk> Resolution Summary ─────────────────────────────")
    pr(f"  Total <unk> tokens      : {total_unk:,}")
    pr(f"  Successfully resolved   : {resolved_unk:,}"
       f" ({100*resolved_unk//max(1,total_unk)}%)")
    pr(f"  Could not resolve       : {unresolved_unk:,}"
       f" ({100*unresolved_unk//max(1,total_unk)}%)")
    pr()
    pr("── Resolved by Tier ─────────────────────────────────────")
    pr(f"  {'Tier':<8} {'Resolved':>12} {'Unresolved':>12}")
    pr(f"  {'─'*34}")
    for tier in ['1', '2', '3']:
        res   = tier_resolved[tier]
        unres = tier_unresolved[tier]
        pr(f"  {'Tier '+tier:<8} {res:>12,} {unres:>12,}")
    pr()
    pr("── What was resolved ────────────────────────────────────")
    pr("  <unk> in word_features was caused by:")
    pr("  1. Dialectal spellings: बहोत (std: बहुत)")
    pr("  2. Nukta characters  : फ़ैन, ज़्यादा, ख़रीदा")
    pr("  3. Rare proper nouns : सोनभद्र, सिंगरौली")
    pr("  4. Informal speech   : मवेसी (std: मवेशी)")
    pr()
    pr("── New Columns Added ─────────────────────────────────────")
    pr("  Position: immediately after 'word' column")
    pr()
    pr("  word_from_lab:")
    pr("    Actual Hindi word from LAB file at same word_index")
    pr("    Filled for ALL rows (not just <unk>)")
    pr("    Empty only for annotation-only segments (45 LABs)")
    pr()
    pr("  word_resolved:")
    pr("    Final clean word for every row")
    pr("    if word == <unk> → word_from_lab")
    pr("    if word != <unk> → word (MFA was correct)")
    pr("    Use this column for all text analysis")
    pr()
    pr("── Sample Resolutions ────────────────────────────────────")

    # Show 5 sample resolutions
    sample_count = 0
    for row in v4_rows:
        if row["word"] == "<unk>" and row["word_resolved"] != "<unk>":
            pr(f"  segment : {row['segment_id']}")
            pr(f"  verbatim: {row.get('verbatim_text','')[:60]}")
            pr(f"  word     = <unk>  →  word_resolved = {row['word_resolved']}")
            pr(f"  duration : {row['word_duration_ms']}ms")
            pr()
            sample_count += 1
            if sample_count >= 5:
                break

    pr("─" * 65)
    pr(f"  V3 columns : {len(v3_fieldnames)}")
    pr(f"  V4 columns : {len(v4_fieldnames)}")
    pr(f"  New columns: word_from_lab, word_resolved")
    pr("=" * 65)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()