#!/usr/bin/env python3
"""
generate_conllu.py — Convert features_B_stanza.csv to CoNLL-U format

Generates CoNLL-U with IC, SC, DepDist in the MISC column.

INPUT : outputs/results/features_B_stanza.csv (614,351 × 14)
OUTPUT: outputs/results/features_B_stanza.conllu

Format per sentence:
  # segment_id = <id>
  # text = <clean_hindi_text>
  ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  _  IC=...|SC=...|DepDist=...

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import csv
import time
from collections import defaultdict

ROOT       = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(ROOT, "outputs", "results")
FEAT_CSV   = os.path.join(RESULTS, "features_B_stanza.csv")
SEG_CSV    = os.path.join(RESULTS,
             "indicvoices_v1_hindi_segments_v4.csv")
OUTPUT     = os.path.join(RESULTS, "features_B_stanza.conllu")


def main():
    print("=" * 60)
    print("  Generate CoNLL-U from features_B_stanza.csv")
    print("=" * 60)
    start = time.time()

    # Load segment text lookup
    print("\n  Loading segment metadata ...")
    seg_text = {}
    with open(SEG_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seg_text[row["segment_id"]] = \
                row.get("clean_hindi_text", "").strip()
    print(f"  Segments loaded: {len(seg_text):,}")

    # Load features grouped by segment (preserving order)
    print("\n  Loading features_B_stanza.csv ...")
    seg_words = defaultdict(list)
    total = 0
    with open(FEAT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seg_words[row["segment_id"]].append(row)
            total += 1
    print(f"  Words loaded : {total:,}")
    print(f"  Segments     : {len(seg_words):,}")

    # Write CoNLL-U
    print(f"\n  Writing CoNLL-U ...")
    sents_written = 0
    words_written = 0

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for seg_id, words in seg_words.items():
            text = seg_text.get(seg_id, "")

            # Comment lines
            f.write(f"# segment_id = {seg_id}\n")
            f.write(f"# text = {text}\n")

            for w in words:
                word_id   = w.get("stanza_word_id", "")
                form      = w.get("word_stanza", "_")
                lemma     = w.get("lemma", "_")
                upos      = w.get("pos_tag", "_") or "_"
                xpos      = w.get("xpos", "_") or "_"
                feats     = w.get("feats", "_") or "_"
                head      = w.get("stanza_head_id", "_")
                deprel    = w.get("dep_relation", "_") or "_"
                ic        = w.get("IC", "")
                sc        = w.get("SC", "")
                dep_dist  = w.get("dep_distance", "")

                # MISC column: IC, SC, DepDist
                misc_parts = []
                if ic:
                    misc_parts.append(f"IC={ic}")
                if sc:
                    misc_parts.append(f"SC={sc}")
                if dep_dist:
                    misc_parts.append(f"DepDist={dep_dist}")
                misc = "|".join(misc_parts) if misc_parts else "_"

                # CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                f.write(f"{word_id}\t{form}\t{lemma}\t{upos}\t{xpos}"
                        f"\t{feats}\t{head}\t{deprel}\t_\t{misc}\n")
                words_written += 1

            f.write("\n")  # blank line between sentences
            sents_written += 1

            if sents_written % 5000 == 0:
                print(f"  [{sents_written:,}/{len(seg_words):,}] "
                      f"{round(100*sents_written/len(seg_words),1)}%")

    sz = round(os.path.getsize(OUTPUT) / 1024 / 1024, 1)
    elapsed = round(time.time() - start, 1)

    print()
    print("=" * 60)
    print("  CoNLL-U GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Sentences written: {sents_written:,}")
    print(f"  Words written    : {words_written:,}")
    print(f"  Size             : {sz} MB")
    print(f"  Time             : {elapsed}s")
    print(f"  Output           : {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()