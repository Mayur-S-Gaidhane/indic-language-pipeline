#!/usr/bin/env python3
"""
merge_all_features.py — Merge all phase outputs into word_features_full.csv

Joins all 5 files on segment_id + word_index:
  word_features_v4.csv       (95 cols) — base
  features_A_simple.csv      ( 4 cols) — Phase A
  features_B_stanza.csv      (12 cols) — Phase B
  features_C_surprisal.csv   ( 2 cols) — Phase C
  features_D_acoustic.csv    (15 cols) — Phase D

OUTPUT:
  word_features_full.csv     (128 cols)

Time: ~10-15 minutes

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import csv
import time

ROOT       = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(ROOT, "outputs", "results")

BASE_CSV   = os.path.join(RESULTS, "word_features_v4.csv")
A_CSV      = os.path.join(RESULTS, "features_A_simple.csv")
B_CSV      = os.path.join(RESULTS, "features_B_stanza.csv")
C_CSV      = os.path.join(RESULTS, "features_C_surprisal.csv")
D_CSV      = os.path.join(RESULTS, "features_D_acoustic.csv")
OUTPUT_CSV = os.path.join(RESULTS, "word_features_full.csv")

# Final column order (128 cols)
FINAL_COLUMNS = [
    # Original 95 cols from word_features_v4.csv
    "segment_id","word_index","word",
    "word_from_lab","word_resolved","word_is_resolved",
    "word_start_sec","word_end_sec","word_duration_ms","phone_count",
    "speaker_id_file","speaker_id_segment","speaker_id_note",
    "gender","age_group","area","state","district",
    "occupation","qualification",
    "scenario","scenario_corrected","task_name","domain",
    "quality_score","quality_score_v4","quality_acoustic_component",
    "quality_linguistic_deduction","dysfluency_density",
    "dataset_tier","is_acoustic_clean",
    "segment_duration_sec","segment_start_sec","segment_end_sec",
    "verbatim_text","verbatim_word_count",
    "normalized_text","normalized_word_count",
    "clean_hindi_text","clean_hindi_word_count",
    "has_code_switch","code_switch_words","cs_word_count",
    "has_dysfluency","dysfluency_types",
    "verbatim_dysfluency_types","normalized_dysfluency_types",
    "dysfluency_count","verbatim_dysfluency_count",
    "normalized_dysfluency_count",
    "has_physiological","physiological_events",
    "verbatim_physiological_events","normalized_physiological_events",
    "physiological_count","verbatim_physiological_count",
    "normalized_physiological_count",
    "has_env_noise","env_noise_events",
    "verbatim_env_noise_events","normalized_env_noise_events",
    "env_noise_count","verbatim_env_noise_count",
    "normalized_env_noise_count",
    "has_quality_annotation","quality_annotations",
    "verbatim_quality_annotations","normalized_quality_annotations",
    "quality_annotation_count","verbatim_quality_annotation_count",
    "normalized_quality_annotation_count",
    "has_paralinguistic","paralinguistic_events",
    "verbatim_paralinguistic_events","normalized_paralinguistic_events",
    "paralinguistic_count","verbatim_paralinguistic_count",
    "normalized_paralinguistic_count",
    "has_action","action_types",
    "verbatim_action_types","normalized_action_types",
    "action_count","verbatim_action_count","normalized_action_count",
    "all_annotations","all_annotation_count",
    "verbatim_all_annotation_count","normalized_all_annotation_count",
    "unknown_annotations","verbatim_unknown_annotations",
    "normalized_unknown_annotations","unknown_annotation_count",
    "verbatim_unknown_annotation_count",
    "normalized_unknown_annotation_count",
    # Phase B: Stanza (12 cols)
    "word_stanza","lemma",
    "stanza_word_id","stanza_head_id",
    "dep_relation","pos_tag","xpos","feats",
    "head_word","dep_distance","IC","SC",
    # Phase A: Simple linguistic (4 cols)
    "word_length_chars","log_duration",
    "word_frequency","log_word_freq",
    # Phase C: Surprisal (2 cols)
    "forward_surprisal","backward_surprisal",
    # Phase D: Acoustic (15 cols)
    "f0_mean_hz","f0_min_hz","f0_max_hz",
    "f0_range_hz","f0_std_hz","voiced_fraction",
    "rms_energy","rms_energy_db",
    "jitter","shimmer","hnr",
    "pause_before_ms","pause_after_ms",
    "speech_rate","acoustic_status",
]

def load_lookup(csv_path, extra_cols):
    """Load a phase CSV into a dict keyed by (segment_id, word_index)."""
    print(f"  Loading {os.path.basename(csv_path)} ...")
    lookup = {}
    count  = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["segment_id"], row["word_index"])
            lookup[key] = {c: row.get(c,"") for c in extra_cols}
            count += 1
    print(f"    Rows: {count:,}")
    return lookup

def main():
    print("="*60)
    print("  MERGE ALL FEATURES")
    print("  word_features_v4 + A + B + C + D")
    print(f"  → word_features_full.csv ({len(FINAL_COLUMNS)} cols)")
    print("="*60)

    start = time.time()

    # Check all input files exist
    missing = []
    for path, name in [
        (BASE_CSV, "word_features_v4.csv"),
        (A_CSV,    "features_A_simple.csv"),
        (B_CSV,    "features_B_stanza.csv"),
        (C_CSV,    "features_C_surprisal.csv"),
        (D_CSV,    "features_D_acoustic.csv"),
    ]:
        if not os.path.exists(path):
            missing.append(name)

    if missing:
        print("\n  ERROR: Missing input files:")
        for m in missing:
            print(f"    ❌ {m}")
        print("\n  Run all phases first before merging.")
        return

    print("\n  All input files found ✅")

    # Load phase lookups
    print("\n  Loading phase outputs ...")
    A_COLS = ["word_length_chars","log_duration",
              "word_frequency","log_word_freq"]
    B_COLS = ["word_stanza","lemma","stanza_word_id","stanza_head_id",
              "dep_relation","pos_tag","xpos","feats",
              "head_word","dep_distance","IC","SC"]
    C_COLS = ["forward_surprisal","backward_surprisal"]
    D_COLS = ["f0_mean_hz","f0_min_hz","f0_max_hz","f0_range_hz",
              "f0_std_hz","voiced_fraction","rms_energy","rms_energy_db",
              "jitter","shimmer","hnr","pause_before_ms","pause_after_ms",
              "speech_rate","acoustic_status"]

    lookup_A = load_lookup(A_CSV, A_COLS)
    lookup_B = load_lookup(B_CSV, B_COLS)
    lookup_C = load_lookup(C_CSV, C_COLS)
    lookup_D = load_lookup(D_CSV, D_COLS)

    # Merge
    print(f"\n  Merging {BASE_CSV} ...")
    total   = 0
    missing_A = missing_B = missing_C = missing_D = 0

    with open(BASE_CSV, encoding="utf-8-sig") as fin, \
         open(OUTPUT_CSV,"w",encoding="utf-8-sig",newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FINAL_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            key = (row["segment_id"], row["word_index"])

            # Merge phase A
            a = lookup_A.get(key)
            if a: row.update(a)
            else:
                missing_A += 1
                for c in A_COLS: row[c] = ""

            # Merge phase B
            b = lookup_B.get(key)
            if b: row.update(b)
            else:
                missing_B += 1
                for c in B_COLS: row[c] = ""

            # Merge phase C
            c_data = lookup_C.get(key)
            if c_data: row.update(c_data)
            else:
                missing_C += 1
                for c in C_COLS: row[c] = ""

            # Merge phase D
            d = lookup_D.get(key)
            if d: row.update(d)
            else:
                missing_D += 1
                for c in D_COLS: row[c] = ""

            writer.writerow(row)
            total += 1

            if total % 100000 == 0:
                print(f"  Written {total:,} rows ...")
                fout.flush()

    sz = round(os.path.getsize(OUTPUT_CSV)/1024/1024/1024, 2)
    elapsed = round(time.time()-start, 1)

    print()
    print("="*60)
    print("  MERGE COMPLETE")
    print("="*60)
    print(f"  Total rows     : {total:,}")
    print(f"  Total columns  : {len(FINAL_COLUMNS)}")
    print(f"  Output size    : {sz} GB")
    print(f"  Time           : {elapsed}s")
    print(f"  Output         : {OUTPUT_CSV}")
    print()
    if missing_A: print(f"  WARNING: {missing_A:,} rows missing Phase A data")
    if missing_B: print(f"  WARNING: {missing_B:,} rows missing Phase B data")
    if missing_C: print(f"  WARNING: {missing_C:,} rows missing Phase C data")
    if missing_D: print(f"  WARNING: {missing_D:,} rows missing Phase D data")
    print()

    # Verify final columns
    print("  Verifying output ...")
    with open(OUTPUT_CSV, encoding="utf-8-sig") as f:
        cols = csv.DictReader(f).fieldnames
    print(f"  Columns verified: {len(cols)} ✅")
    print("="*60)

if __name__ == "__main__":
    main()