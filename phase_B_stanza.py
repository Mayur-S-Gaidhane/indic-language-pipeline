#!/usr/bin/env python3
"""
phase_B_stanza.py — Phase B: Stanza Dependency Features (FIXED v2)

IC/SC per Dammalapati, Rajkumar, Ranjan & Agarwal (SCiL 2021)
and verified against Prof's dlt.py:

  IC = baseline 1 + sum of intervening NON-PUNCT words,
       assigned to the LATER token in each dependency (both directions).
       PUNCT tokens get IC = 0.

  SC = baseline 1 + count of incomplete dependencies (d <= i < h).
       PUNCT tokens get SC = 0.

Verified against verify_ic_sc.py — 100% match with Prof's dlt.py.

INPUT : outputs/results/word_features_v4.csv (614,351 x 95)
        outputs/results/indicvoices_v1_hindi_segments_v4.csv
OUTPUT: outputs/results/features_B_stanza.csv (614,351 x 14)
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

EXCLUDED_UPOS = {"PUNCT"}


def compute_ic_sc(words):
    """
    IC/SC matching Prof's dlt.py and SCiL 2021 paper (Section 2.2).

    IC (Integration Cost):
        For each dependency arc (dep, head):
          - lower = min(dep.id, head.id)
          - upper = max(dep.id, head.id)
          - Count intervening NON-PUNCT tokens between lower and upper
          - Add that count to IC of the LATER token (upper)
        Baseline = 1 for every non-PUNCT token.
        PUNCT tokens = 0.

    SC (Storage Cost):
        For each position i:
          SC(i) = 1 + count of dependencies where dep.id <= i < head.id
        Baseline = 1 for every non-PUNCT token.
        PUNCT tokens = 0.

    Both match Prof's dlt.py verified output.
    """
    n = len(words)
    ic_list = [0] * n
    sc_list = [0] * n

    # Build id-to-index map (word.id is 1-based)
    id_to_idx = {}
    for i in range(n):
        id_to_idx[words[i].id] = i

    # ── IC ────────────────────────────────────────────────────
    # Baseline 1 for non-PUNCT
    for i in range(n):
        if words[i].upos not in EXCLUDED_UPOS:
            ic_list[i] = 1

    # For each dependency, add intervening non-PUNCT count to later token
    for i in range(n):
        dep_id  = words[i].id       # 1-based
        head_id = words[i].head     # 1-based, 0 = root
        if head_id == 0:
            continue
        if head_id not in id_to_idx:
            continue

        lower = min(dep_id, head_id)
        upper = max(dep_id, head_id)
        later_idx = id_to_idx[upper]

        # Count intervening non-PUNCT tokens
        intervening = 0
        for j in range(n):
            wid = words[j].id
            if lower < wid < upper:
                if words[j].upos not in EXCLUDED_UPOS:
                    intervening += 1

        ic_list[later_idx] += intervening

    # Zero out PUNCT
    for i in range(n):
        if words[i].upos in EXCLUDED_UPOS:
            ic_list[i] = 0

    # ── SC ────────────────────────────────────────────────────
    # Collect valid dependencies (exclude root)
    valid_ids = set(w.id for w in words)
    dependencies = []
    for w in words:
        if w.head != 0 and w.head in valid_ids:
            dependencies.append((w.id, w.head))

    for i in range(n):
        if words[i].upos in EXCLUDED_UPOS:
            sc_list[i] = 0
            continue
        word_id = words[i].id
        unresolved = 0
        for (d, h) in dependencies:
            if d <= word_id < h:
                unresolved += 1
        sc_list[i] = 1 + unresolved

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
    print("  PHASE B: Stanza Dependency Features")
    print("  IC/SC per SCiL 2021 + Prof's dlt.py (FIXED v2)")
    print("  IC = 1 + intervening non-PUNCT, assigned to later token")
    print("  SC = 1 + incomplete deps (d <= i < h)")
    print("  PUNCT zeroed out for both IC and SC")
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
    print("  PHASE B COMPLETE (IC/SC FIXED v2 — matches dlt.py)")
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