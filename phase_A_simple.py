#!/usr/bin/env python3
"""
phase_A_simple.py — Phase A: Simple Linguistic Features

Computes 4 simple features from word_features_v4.csv:
  word_length_chars  = len(word_resolved)
  log_duration       = log(word_duration_ms)
  word_frequency     = corpus count from full dataset
  log_word_freq      = log10(word_frequency)

INPUT : outputs/results/word_features_v4.csv (614,351 × 95)
OUTPUT: outputs/results/features_A_simple.csv (614,351 × 6)
        columns: segment_id, word_index +4 features

Time: ~5 minutes

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import csv
import math
import time
from collections import Counter

ROOT       = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(ROOT, "outputs", "results")
INPUT_CSV  = os.path.join(RESULTS, "word_features_v4.csv")
OUTPUT_CSV = os.path.join(RESULTS, "features_A_simple.csv")

def main():
    print("="*60)
    print("  PHASE A: Simple Linguistic Features")
    print("  word_length_chars, log_duration")
    print("  word_frequency, log_word_freq")
    print("="*60)

    start = time.time()

    print(f"\n  Loading {INPUT_CSV} ...")
    rows = []
    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  Rows loaded: {len(rows):,}")

    # Compute word frequency from full corpus
    print("\n  Computing word frequencies ...")
    freq = Counter()
    for row in rows:
        wr = (row.get("word_resolved","") or "").strip().lower()
        if wr and wr not in ("<unk>","sp","sil","spn"):
            freq[wr] += 1
    print(f"  Unique word types : {len(freq):,}")
    print(f"  Total tokens      : {sum(freq.values()):,}")

    # Write output
    print(f"\n  Writing {OUTPUT_CSV} ...")
    FIELDNAMES = [
        "segment_id","word_index",
        "word_length_chars","log_duration",
        "word_frequency","log_word_freq",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, row in enumerate(rows):
            wr  = (row.get("word_resolved","") or "").strip()
            dur = row.get("word_duration_ms","")

            try:
                d = float(dur)
                log_dur = round(math.log(d) if d > 0 else 0, 6)
            except:
                log_dur = ""

            wfreq = freq.get(wr.lower(), 1)

            writer.writerow({
                "segment_id"      : row["segment_id"],
                "word_index"      : row["word_index"],
                "word_length_chars": len(wr) if wr else 0,
                "log_duration"    : log_dur,
                "word_frequency"  : wfreq,
                "log_word_freq"   : round(math.log10(wfreq), 4),
            })

            if (i+1) % 100000 == 0:
                print(f"  [{i+1:,}/{len(rows):,}]")

    sz = round(os.path.getsize(OUTPUT_CSV)/1024/1024, 1)
    elapsed = round(time.time()-start, 1)

    print()
    print("="*60)
    print("  PHASE A COMPLETE")
    print("="*60)
    print(f"  Rows    : {len(rows):,}")
    print(f"  Columns : {len(FIELDNAMES)}")
    print(f"  Size    : {sz} MB")
    print(f"  Time    : {elapsed}s")
    print(f"  Output  : {OUTPUT_CSV}")
    print("="*60)

if __name__ == "__main__":
    main()