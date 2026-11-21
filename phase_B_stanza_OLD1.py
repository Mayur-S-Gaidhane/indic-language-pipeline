#!/usr/bin/env python3
"""
phase_B_stanza.py — Phase B: Stanza Dependency Features (FIXED)

IC/SC per Dammalapati, Rajkumar, Ranjan & Agarwal (SCiL 2021):
  IC = minimum cost 1 + sum of INTERVENING-WORD distances (backward)
  SC = minimum cost 1 + count of INCOMPLETE dependencies (forward)

Verified against paper's Fig (page 93):
  "My dog also likes eating sausage"
  IC = [1, 1, 1, 2, 1, 1]
  SC = [2, 2, 3, 1, 1, 1]

Parses clean_hindi_text with Stanza and extracts:
  word_stanza, lemma, stanza_word_id, stanza_head_id,
  dep_relation, pos_tag, xpos, feats,
  head_word, dep_distance, IC, SC

INPUT : outputs/results/word_features_v4.csv (614,351 × 95)
        outputs/results/indicvoices_v1_hindi_segments_v4.csv
OUTPUT: outputs/results/features_B_stanza.csv (614,351 × 14)
        columns: segment_id, word_index + 12 stanza features

Time: ~3-4 hours for 28,302 segments

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import csv
import time
from collections import defaultdict

ROOT       = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(ROOT, "outputs", "results")
WORD_CSV   = os.path.join(RESULTS, "word_features_v4.csv")
SEG_CSV    = os.path.join(RESULTS,
             "indicvoices_v1_hindi_segments_v4.csv")
OUTPUT_CSV = os.path.join(RESULTS, "features_B_stanza.csv")

FIELDNAMES = [
    "segment_id","word_index",
    "word_stanza","lemma",
    "stanza_word_id","stanza_head_id",
    "dep_relation","pos_tag","xpos","feats",
    "head_word","dep_distance","IC","SC",
]


def compute_ic_sc(words):
    """
    IC/SC computation — Dammalapati, Rajkumar, Ranjan & Agarwal (SCiL 2021)
    "Effects of Duration, Locality, and Surprisal in Speech Disfluency
     Prediction in English Spontaneous Speech", Section 2.2.

    IC (Integration Cost) at word_i:
        Minimum cost of 1 (for the word itself) PLUS the sum of
        dependency lengths for all deps COMPLETED at word_i.
        Dependency length = number of INTERVENING words between
        dependent and head (i.e. |head_pos - dep_pos| - 1).
        Backward-looking: only earlier words whose head IS word_i.

    SC (Storage Cost) at word_i:
        Minimum cost of 1 (for the word itself) PLUS the count of
        incomplete dependencies — words seen so far (1..i) whose
        head has NOT yet appeared (head_pos > i).
        Forward-looking.

    Verified: "My dog also likes eating sausage"
      IC = [1, 1, 1, 2, 1, 1]
      SC = [2, 2, 3, 1, 1, 1]
    (matches paper Fig on page 93)
    """
    n = len(words)
    ic_list = [0] * n
    sc_list = [0] * n

    for i in range(n):
        word_pos = i + 1  # 1-based position (Stanza convention)

        # ── IC: Integration Cost ──────────────────────────────
        # Minimum cost 1 + sum of intervening-word distances
        # for dependencies completed at this word.
        ic = 1  # minimum cost per Gibson (2000) / SCiL paper
        for j in range(i):  # j < i: only earlier words
            dep_pos = j + 1
            dep_head = words[j].head
            if dep_head == word_pos:
                # Intervening words = |head - dep| - 1
                ic += abs(dep_pos - word_pos) - 1
        ic_list[i] = ic

        # ── SC: Storage Cost ──────────────────────────────────
        # Minimum cost 1 + count of incomplete dependencies.
        sc = 1  # minimum cost per Gibson (2000) / SCiL paper
        for j in range(i + 1):  # words seen so far (including current)
            dep_head = words[j].head
            if dep_head == 0:
                # HEAD is ROOT — always considered resolved
                continue
            if dep_head > word_pos:
                # Head hasn't appeared yet — incomplete dependency
                sc += 1
        sc_list[i] = sc

    return ic_list, sc_list


def empty_stanza_row(seg_id, word_index):
    return {
        "segment_id"    : seg_id,
        "word_index"    : word_index,
        "word_stanza"   : "", "lemma"          : "",
        "stanza_word_id": "", "stanza_head_id" : "",
        "dep_relation"  : "", "pos_tag"        : "",
        "xpos"          : "", "feats"          : "",
        "head_word"     : "", "dep_distance"   : "",
        "IC"            : "", "SC"             : "",
    }

def main():
    print("="*60)
    print("  PHASE B: Stanza Dependency Features (SCiL 2021 IC/SC)")
    print("  IC = min cost 1 + sum of intervening-word distances (backward)")
    print("  SC = min cost 1 + count of incomplete deps (forward)")
    print("  Verified: 'My dog also likes eating sausage'")
    print("  IC=[1,1,1,2,1,1], SC=[2,2,3,1,1,1]")
    print("="*60)

    start = time.time()

    # Check stanza
    try:
        import stanza
    except ImportError:
        print("ERROR: stanza not installed")
        print("Run: pip install stanza")
        return

    # Load segment text lookup
    print(f"\n  Loading segment metadata ...")
    seg_text = {}
    with open(SEG_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seg_text[row["segment_id"]] = \
                row.get("clean_hindi_text","").strip()
    print(f"  Segments loaded: {len(seg_text):,}")

    # Load word rows grouped by segment
    print(f"\n  Loading word_features_v4.csv ...")
    seg_words = defaultdict(list)
    total_words = 0
    with open(WORD_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seg_words[row["segment_id"]].append({
                "segment_id": row["segment_id"],
                "word_index": row["word_index"],
            })
            total_words += 1
    print(f"  Words loaded : {total_words:,}")
    print(f"  Segments     : {len(seg_words):,}")

    # Load Stanza
    print(f"\n  Loading Stanza Hindi pipeline ...")
    nlp = stanza.Pipeline(
        lang="hi",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        verbose=False
    )
    print(f"  Stanza loaded")

    # Process
    print(f"\n  Processing {len(seg_words):,} segments ...")

    done_segs  = 0
    done_words = 0
    errors     = 0
    total_segs = len(seg_words)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for seg_id, word_list in seg_words.items():
            text = seg_text.get(seg_id, "").strip()

            if not text:
                for w in word_list:
                    writer.writerow(
                        empty_stanza_row(seg_id, w["word_index"]))
                done_words += len(word_list)
                done_segs  += 1
                continue

            try:
                tokens = text.split()
                doc    = nlp([tokens])
                sent   = doc.sentences[0]
                sw     = sent.words
                ic_list, sc_list = compute_ic_sc(sw)

                for wi, w in enumerate(word_list):
                    if wi < len(sw):
                        s  = sw[wi]
                        hw = sw[s.head-1].text if s.head > 0 else "ROOT"
                        dd = abs(s.head - s.id) if s.head > 0 else 0
                        writer.writerow({
                            "segment_id"    : seg_id,
                            "word_index"    : w["word_index"],
                            "word_stanza"   : s.text,
                            "lemma"         : s.lemma or s.text,
                            "stanza_word_id": s.id,
                            "stanza_head_id": s.head,
                            "dep_relation"  : s.deprel or "",
                            "pos_tag"       : s.upos or "",
                            "xpos"          : s.xpos or "",
                            "feats"         : s.feats or "",
                            "head_word"     : hw,
                            "dep_distance"  : dd,
                            "IC"            : ic_list[wi],
                            "SC"            : sc_list[wi],
                        })
                    else:
                        writer.writerow(
                            empty_stanza_row(seg_id, w["word_index"]))

            except Exception as e:
                errors += 1
                for w in word_list:
                    writer.writerow(
                        empty_stanza_row(seg_id, w["word_index"]))

            done_words += len(word_list)
            done_segs  += 1

            if done_segs % 500 == 0:
                pct     = round(100*done_segs/total_segs, 1)
                elapsed = round(time.time()-start, 0)
                eta     = round(elapsed/done_segs*(total_segs-done_segs),0)
                print(f"  [{done_segs:,}/{total_segs:,}] {pct}% "
                      f"| elapsed={int(elapsed)}s ETA={int(eta)}s"
                      f" | errors={errors}")

    sz = round(os.path.getsize(OUTPUT_CSV)/1024/1024, 1)
    elapsed = round(time.time()-start, 1)

    print()
    print("="*60)
    print("  PHASE B COMPLETE (FIXED IC/SC)")
    print("="*60)
    print(f"  Words processed: {done_words:,}")
    print(f"  Segs processed : {done_segs:,}")
    print(f"  Errors         : {errors:,}")
    print(f"  Columns        : {len(FIELDNAMES)}")
    print(f"  Size           : {sz} MB")
    print(f"  Time           : {elapsed}s ({round(elapsed/3600,1)}h)")
    print(f"  Output         : {OUTPUT_CSV}")
    print("="*60)

if __name__ == "__main__":
    main()