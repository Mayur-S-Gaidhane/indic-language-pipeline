#!/usr/bin/env python3
"""
sync_word_features_v4.py  — FIXED VERSION

Reads word_features_v4.csv (44 cols) and segment CSV V4 (129 cols).
Produces a fully synced word_features_v4.csv with COMPLETE column set
matching the segment CSV V4 structure.

Strategy:
  Define complete desired column order explicitly.
  For each column: use existing value from word_features
                   OR join from segment CSV V4 on segment_id.
"""

import os, sys, csv, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config

WORD_V4  = os.path.join(config.RESULTS_DIR, "word_features_v4.csv")
SEG_V4   = os.path.join(config.RESULTS_DIR,
            "indicvoices_v1_hindi_segments_v4.csv")
REPORT   = os.path.join(config.RESULTS_DIR,
            "word_features_v4_report.txt")

# ── Complete desired column order ─────────────────────────────
# Groups: Acoustic | Speaker | Segment | Text | Annotations
FINAL_COLUMNS = [
    # GROUP 1: Acoustic (from TextGrid / add_word_lab_mapping)
    "segment_id",
    "word_index",
    "word",
    "word_from_lab",
    "word_resolved",
    "word_is_resolved",          # 1 if was <unk> and replaced, 0 otherwise
    "word_start_sec",
    "word_end_sec",
    "word_duration_ms",
    "phone_count",

    # GROUP 2: Speaker (from segment CSV)
    "speaker_id_file",
    "speaker_id_segment",
    "speaker_id_note",           # V4 new
    "gender",
    "age_group",
    "area",
    "state",
    "district",
    "occupation",
    "qualification",

    # GROUP 3: Segment metadata
    "scenario",
    "scenario_corrected",
    "task_name",
    "domain",
    "quality_score",
    "quality_score_v4",          # V4 new
    "quality_acoustic_component",# V4 new
    "quality_linguistic_deduction",# V4 new
    "dysfluency_density",        # V4 new
    "dataset_tier",
    "is_acoustic_clean",
    "segment_duration_sec",
    "segment_start_sec",
    "segment_end_sec",

    # GROUP 4: Text content
    "verbatim_text",
    "verbatim_word_count",
    "normalized_text",
    "normalized_word_count",
    "clean_hindi_text",          # V4 new
    "clean_hindi_word_count",    # V4 new

    # GROUP 5: Code switching
    "has_code_switch",
    "code_switch_words",
    "cs_word_count",

    # GROUP 6: Dysfluency
    "has_dysfluency",
    "dysfluency_types",
    "verbatim_dysfluency_types",   # V4 new
    "normalized_dysfluency_types", # V4 new
    "dysfluency_count",
    "verbatim_dysfluency_count",   # V4 new
    "normalized_dysfluency_count", # V4 new

    # GROUP 7: Physiological
    "has_physiological",
    "physiological_events",
    "verbatim_physiological_events",  # V4 new
    "normalized_physiological_events",# V4 new
    "physiological_count",
    "verbatim_physiological_count",   # V4 new
    "normalized_physiological_count", # V4 new

    # GROUP 8: Environmental noise
    "has_env_noise",
    "env_noise_events",
    "verbatim_env_noise_events",   # V4 new
    "normalized_env_noise_events", # V4 new
    "env_noise_count",
    "verbatim_env_noise_count",    # V4 new
    "normalized_env_noise_count",  # V4 new

    # GROUP 9: Quality annotations
    "has_quality_annotation",
    "quality_annotations",
    "verbatim_quality_annotations",   # V4 new
    "normalized_quality_annotations", # V4 new
    "quality_annotation_count",
    "verbatim_quality_annotation_count",  # V4 new
    "normalized_quality_annotation_count",# V4 new

    # GROUP 10: Paralinguistic
    "has_paralinguistic",
    "paralinguistic_events",
    "verbatim_paralinguistic_events",  # V4 new
    "normalized_paralinguistic_events",# V4 new
    "paralinguistic_count",
    "verbatim_paralinguistic_count",   # V4 new
    "normalized_paralinguistic_count", # V4 new

    # GROUP 11: Action
    "has_action",
    "action_types",
    "verbatim_action_types",   # V4 new
    "normalized_action_types", # V4 new
    "action_count",
    "verbatim_action_count",   # V4 new
    "normalized_action_count", # V4 new

    # GROUP 12: All annotations
    "all_annotations",
    "all_annotation_count",
    "verbatim_all_annotation_count",  # V4 new
    "normalized_all_annotation_count",# V4 new
    "unknown_annotations",
    "verbatim_unknown_annotations",   # V4 new
    "normalized_unknown_annotations", # V4 new
    "unknown_annotation_count",
    "verbatim_unknown_annotation_count",  # V4 new
    "normalized_unknown_annotation_count",# V4 new
]


def main():
    print("=" * 65)
    print("  SYNC WORD FEATURES V4 — COMPLETE COLUMN SET")
    print("=" * 65)
    print()

    for path in [WORD_V4, SEG_V4]:
        if not os.path.exists(path):
            print(f"  ERROR: Not found: {path}")
            return

    # ── Load segment V4 lookup ────────────────────────────────
    print("  Loading segment V4 CSV ...")
    seg_lookup = {}
    with open(SEG_V4, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            seg_lookup[row["segment_id"]] = row
    print(f"  Segment V4: {len(seg_lookup):,} rows")

    # ── Load current word_features_v4 ─────────────────────────
    print("  Loading word_features_v4.csv ...")
    with open(WORD_V4, encoding="utf-8-sig") as f:
        reader       = csv.DictReader(f)
        existing_cols = set(reader.fieldnames)
        word_rows    = list(reader)
    print(f"  Word V4: {len(word_rows):,} rows, "
          f"{len(existing_cols)} columns")

    # ── Determine which columns come from where ───────────────
    from_word_csv = existing_cols
    from_seg_csv  = set(FINAL_COLUMNS) - from_word_csv
    print(f"  Final columns  : {len(FINAL_COLUMNS)}")
    print(f"  From word CSV  : {len(from_word_csv & set(FINAL_COLUMNS))}")
    print(f"  From segment V4: {len(from_seg_csv)}")
    print()

    # Show what is being added
    new_cols = [c for c in FINAL_COLUMNS if c not in existing_cols]
    print(f"  New columns being added ({len(new_cols)}):") 
    for c in new_cols:
        print(f"    + {c}")
    print()

    # ── Process rows ──────────────────────────────────────────
    print("  Processing rows ...")
    start = time.time()
    seg_not_found = 0
    v4_rows = []

    for idx, row in enumerate(word_rows):
        if (idx + 1) % 100000 == 0:
            print(f"  [{idx+1:>7,}/{len(word_rows):,}]")

        seg_id  = row.get("segment_id", "")
        seg_row = seg_lookup.get(seg_id, {})

        if not seg_row:
            seg_not_found += 1

        # Build output row with all final columns
        out = {}
        for col in FINAL_COLUMNS:
            if col == "word_is_resolved":
                # 1 if original word was <unk> and got replaced
                # 0 if MFA aligned the word correctly
                original_word = row.get("word", "")
                out[col] = 1 if original_word == "<unk>" else 0
            elif col in row and row[col] != "":
                out[col] = row[col]
            elif col in seg_row:
                out[col] = seg_row[col]
            else:
                out[col] = ""

        v4_rows.append(out)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Segment IDs not found: {seg_not_found:,}")

    # ── Write output ──────────────────────────────────────────
    print(f"\n  Writing {WORD_V4} ...")
    with open(WORD_V4, "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=FINAL_COLUMNS,
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(v4_rows)

    size_mb = os.path.getsize(WORD_V4) / 1024 / 1024
    print(f"  File size : {size_mb:.1f} MB")
    print(f"  Columns   : {len(FINAL_COLUMNS)}")

    # ── Verification ──────────────────────────────────────────
    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr()
    pr("=" * 65)
    pr("  VERIFICATION REPORT")
    pr("=" * 65)
    pr(f"  Total rows     : {len(v4_rows):,}")
    pr(f"  Total columns  : {len(FINAL_COLUMNS)}")
    pr(f"  New cols added : {len(new_cols)}")
    pr()

    # word_is_resolved stats
    total_replaced   = sum(1 for r in v4_rows if str(r.get("word_is_resolved","")) == "1")
    total_correct    = sum(1 for r in v4_rows if str(r.get("word_is_resolved","")) == "0")
    pr("── word_is_resolved Summary ─────────────────────────────")
    pr(f"  word_is_resolved = 1 (was <unk>, now fixed) : {total_replaced:,}")
    pr(f"  word_is_resolved = 0 (MFA correct)          : {total_correct:,}")
    pr(f"  Resolution rate  : {100*total_replaced//max(1,total_replaced+total_correct)}%")
    pr()

    # Spot check — phone_vibrating → has_env_noise
    sample_env = next(
        (r for r in v4_rows
         if 'phone_vibrating' in r.get('verbatim_text','')), None)
    if sample_env:
        pr("  ✅ phone_vibrating test:")
        pr(f"     has_env_noise             = {sample_env.get('has_env_noise')}")
        pr(f"     env_noise_events          = {sample_env.get('env_noise_events','')[:40]}")
        pr(f"     verbatim_env_noise_count  = {sample_env.get('verbatim_env_noise_count')}")
        pr(f"     verbatim_env_noise_events = {sample_env.get('verbatim_env_noise_events','')[:40]}")

    # Spot check — throat_clearing → has_physiological
    sample_phy = next(
        (r for r in v4_rows
         if 'throat_clearing' in r.get('verbatim_text','')), None)
    if sample_phy:
        pr()
        pr("  ✅ throat_clearing test:")
        pr(f"     has_physiological               = {sample_phy.get('has_physiological')}")
        pr(f"     physiological_events            = {sample_phy.get('physiological_events','')[:40]}")
        pr(f"     verbatim_physiological_count    = {sample_phy.get('verbatim_physiological_count')}")
        pr(f"     verbatim_physiological_events   = {sample_phy.get('verbatim_physiological_events','')[:40]}")

    # Spot check — word_resolved
    sample_unk = next(
        (r for r in v4_rows
         if r.get('word_resolved','') not in ('','<unk>')
         and r.get('word','') == '<unk>'), None)
    if sample_unk:
        pr()
        pr("  ✅ <unk> resolution test:")
        pr(f"     word          = {sample_unk.get('word')}")
        pr(f"     word_from_lab = {sample_unk.get('word_from_lab')}")
        pr(f"     word_resolved = {sample_unk.get('word_resolved')}")

    # Spot check — clean_hindi_text
    sample_cs = next(
        (r for r in v4_rows
         if r.get('cs_word_count','0') not in ('','0')
         and r.get('clean_hindi_text','')), None)
    if sample_cs:
        pr()
        pr("  ✅ clean_hindi_text test:")
        pr(f"     verbatim  : {sample_cs.get('verbatim_text','')[:55]}")
        pr(f"     normalized: {sample_cs.get('normalized_text','')[:55]}")
        pr(f"     clean     : {sample_cs.get('clean_hindi_text','')[:55]}")

    pr()
    pr("  Final column order:")
    for i, c in enumerate(FINAL_COLUMNS, 1):
        marker = " ← NEW" if c not in existing_cols else ""
        pr(f"    {i:3d}. {c}{marker}")
    pr()
    pr("=" * 65)

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {REPORT}")


if __name__ == "__main__":
    main()