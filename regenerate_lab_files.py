#!/usr/bin/env python3
"""
regenerate_lab_files.py

PURPOSE
-------
Fix the two systematic causes of false OOV words found by
check_mfa_oov.py, WITHOUT re-cutting any audio:

    Pattern 1 (~80% of OOV types): Hyphenated reduplications /
        compounds, e.g. "अलग-अलग", "धीरे-धीरे", "दो-तीन".
        These are two perfectly valid dictionary words joined by
        a hyphen. MFA's tokenizer treats the hyphen as part of
        the word, so it can never match either half. Splitting
        on the hyphen turns one OOV "word" into two in-vocabulary
        words (e.g. "अलग-अलग" -> "अलग अलग").

    Pattern 2 (~15% of OOV types): Stray punctuation left
        attached to a word, e.g. "है," "है?" or a standalone "?".
        These were not fully stripped when the original LAB
        files were written in build_segment_dataset_v3.py
        (only bracketed annotation tags like [horn] were
        removed there, not general punctuation).

    Pattern 3 (found after first fix pass): Malformed bracket
        tags with internal whitespace, e.g. "[ inhaling ]",
        left orphaned fragments like "[inhaling" / "inhaling]"
        when the original regex required brackets immediately
        adjacent to the tag text.

    Pattern 4 (found after first fix pass): Devanagari sentence-
        final punctuation "।" (purna viram, U+0964), e.g. "है।",
        "करें।" -- not covered by the original ASCII-only
        punctuation strip.

    Pattern 5 (found after first fix pass): Invisible zero-width
        Unicode characters embedded inside words, e.g.
        "अच्\u200dछा" (ZERO WIDTH JOINER mid-word) or
        "\u200b\u200bथा" (ZERO WIDTH SPACE before a word).
        Invisible on screen, but break exact dictionary matching.
        Likely an artifact of the original IndicVoices annotation
        process (e.g. copy-pasting text from a source with hidden
        formatting characters).

    Pattern 6 (found after second fix pass): Annotation tags (or
        parenthetical asides) split across two ADJACENT verbatim
        segments by the original IndicVoices annotators, e.g. one
        segment ends with "...[inhaling" (opening bracket, no
        closing bracket -- it is in the next segment) and the
        following segment begins with "inhaling]..." (closing
        bracket, no opening bracket -- it was in the previous
        segment). Because each segment's .lab file is generated
        independently, this can never be resolved by looking at
        a single segment's text alone; the only safe fix is to
        strip any token that contains an unmatched "[", "(", "]"
        or ")" together with the rest of that token, since such a
        token can never be a real Hindi word. Also added a few
        additional stray punctuation characters found at this
        stage: backtick, pipe "|", forward slash "/".

WHY REGENERATE ONLY .lab FILES
-------------------------------
The WAV audio files cut by build_segment_dataset_v3.py are
unaffected by this fix -- only the transcript text fed to MFA
needs to change. Re-cutting ~29,231 WAV files takes roughly 80
minutes; rewriting ~29,231 small text files takes a few seconds.
This script reads verbatim_text directly from the already-built
V3 CSV (the single source of truth for transcripts) and overwrites
each segment's existing .lab file in place at segment_lab_path.

CLEANING RULES APPLIED
-----------------------
1. Remove bracketed annotation tags, e.g. [horn], [stammers]
   (same as the original build script -- repeated here so this
   script is self-contained and idempotent).
2. Replace hyphens with a space, splitting compounds/
   reduplications into separate, dictionary-valid words.
3. Remove stray punctuation characters that are not part of
   Devanagari script: , ? ! . : ; " '
   (Devanagari's own visarga ः and other native marks are NOT
   touched -- only ASCII punctuation marks are removed.)
4. Collapse any resulting multiple spaces and strip leading/
   trailing whitespace.

USAGE
-----
    python regenerate_lab_files.py

Run from the project root (the directory containing
mfa_segments/ and outputs/results/indicvoices_v1_hindi_segments_v3.csv).

After running this script, re-run check_mfa_oov.py and then
`mfa validate` to confirm the OOV rate has dropped before
proceeding to full `mfa align`.

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
Project   : MTP-2 — Acoustic-Linguistic Modelling of Cognitive
            Constraints in Hindi Conversational Speech
"""

import os
import re
import csv

# ── Configuration ──────────────────────────────────────────────
INPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "results", "indicvoices_v1_hindi_segments_v3.csv")

# ASCII punctuation to strip outright (no replacement character).
# Devanagari-native marks that carry phonetic meaning (e.g. ः
# visarga) are intentionally NOT included here. "।" (U+0964,
# purna viram / Devanagari full stop) IS included -- it is
# punctuation, not a phoneme, and was found causing OOV words
# like "है।" and "करें।" in the V3 corpus. Backtick, pipe, and
# forward slash were added after a second OOV check found stray
# occurrences (e.g. "के`भी", "|", "/").
PUNCT_PATTERN = re.compile(
    r'[,?!.;:"\'\u2018\u2019\u201c\u201d\u0964\u0965$`|/]'
)

# Zero-width Unicode characters that are invisible on screen but
# break exact string/dictionary matching when embedded inside a
# word, e.g. "अच्\u200dछा" (ZERO WIDTH JOINER inserted mid-word)
# or "\u200b\u200b\u200bथा" (ZERO WIDTH SPACE before a word).
# Found in the V3 corpus, likely introduced during the original
# IndicVoices annotation/copy-paste process.
ZERO_WIDTH_PATTERN = re.compile(
    r'[\u200b\u200c\u200d\ufeff]'
)

# Bracketed annotation tags, e.g. [horn], [stammers], [inhaling].
# Allows optional whitespace immediately inside the brackets
# (e.g. "[ inhaling ]") which the original pattern missed,
# leaving behind orphaned "[inhaling" / "inhaling]" fragments.
BRACKET_TAG_PATTERN = re.compile(r'\[\s*[^\]]*?\s*\]')

# Unmatched/orphaned bracket or parenthesis fragments that arise
# when an annotation tag (or a parenthetical aside) was split
# across two adjacent verbatim segments by the original
# IndicVoices annotators -- e.g. one segment ends with "...
# [inhaling" (opening bracket, no closing bracket in this
# segment because it appears in the NEXT segment) and the next
# segment begins with "inhaling] ..." (closing bracket, no
# opening bracket because it was in the PREVIOUS segment).
# Because each segment is processed independently, a single
# segment can never resolve this -- the safest fix is to strip
# the bracket/parenthesis character together with the rest of
# its own token (up to the next whitespace), since a token
# containing an unmatched bracket is never a real Hindi word.
UNMATCHED_OPEN_BRACKET_PATTERN = re.compile(r'[\[(]\s*\S*')
UNMATCHED_CLOSE_BRACKET_PATTERN = re.compile(r'\S*\s*[\])]')


def clean_lab_text(verbatim_text):
    """
    Apply all cleaning rules to a single segment's verbatim text
    and return the text that should be written to its .lab file.
    """
    text = verbatim_text

    # 1. Remove complete bracketed annotation tags (handles
    #    stray internal whitespace, e.g. "[ inhaling ]")
    text = BRACKET_TAG_PATTERN.sub('', text)

    # 2. Remove leftover unmatched bracket/parenthesis fragments
    #    caused by an annotation tag being split across two
    #    adjacent verbatim segments (see comment above the
    #    pattern definitions for why this can't be fully
    #    resolved any other way at the single-segment level).
    text = UNMATCHED_OPEN_BRACKET_PATTERN.sub('', text)
    text = UNMATCHED_CLOSE_BRACKET_PATTERN.sub('', text)

    # 3. Strip invisible zero-width Unicode characters that may
    #    be embedded inside words
    text = ZERO_WIDTH_PATTERN.sub('', text)

    # 4. Split hyphenated compounds/reduplications into separate
    #    words by replacing the hyphen with a space.
    text = text.replace('-', ' ')

    # 5. Remove stray punctuation (ASCII + Devanagari purna viram)
    text = PUNCT_PATTERN.sub('', text)

    # 6. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def main():
    print("=" * 65)
    print("  REGENERATE LAB FILES — Fix OOV from hyphens/punctuation")
    print("=" * 65)

    if not os.path.exists(INPUT_CSV):
        print(f"\n  ERROR: CSV not found at {INPUT_CSV}")
        print("  Run build_segment_dataset_v3.py first.")
        return

    rows_total = 0
    rows_updated = 0
    rows_skipped_no_lab = 0
    rows_skipped_no_change = 0
    example_changes = []

    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1

            lab_path = row.get("segment_lab_path", "")
            is_mfa_ready = row.get("is_mfa_ready", "False") == "True"

            if not lab_path or not is_mfa_ready:
                rows_skipped_no_lab += 1
                continue

            verbatim_text = row.get("verbatim_text", "")
            new_lab_text = clean_lab_text(verbatim_text)

            # Read existing lab file content for comparison
            old_lab_text = ""
            if os.path.exists(lab_path):
                with open(lab_path, encoding="utf-8") as lf:
                    old_lab_text = lf.read().strip()

            if old_lab_text == new_lab_text:
                rows_skipped_no_change += 1
                continue

            os.makedirs(os.path.dirname(lab_path), exist_ok=True)
            with open(lab_path, "w", encoding="utf-8") as lf:
                lf.write(new_lab_text)

            rows_updated += 1
            if len(example_changes) < 10 and old_lab_text != new_lab_text:
                example_changes.append((lab_path, old_lab_text,
                                         new_lab_text))

            if rows_total % 5000 == 0:
                print(f"  Processed {rows_total:,} rows  "
                      f"(updated={rows_updated:,})")

    print()
    print(f"  Total rows in CSV       : {rows_total:,}")
    print(f"  Rows without a LAB file : {rows_skipped_no_lab:,} "
          f"(text-only / not MFA-ready)")
    print(f"  LAB files unchanged     : {rows_skipped_no_change:,}")
    print(f"  LAB files updated       : {rows_updated:,}")

    if example_changes:
        print()
        print("  Example changes:")
        for path, old, new in example_changes:
            short_path = os.path.basename(path)
            print(f"    {short_path}")
            print(f"      before: {old!r}")
            print(f"      after : {new!r}")

    print()
    print("=" * 65)
    print("  NEXT STEPS")
    print("=" * 65)
    print("  1. python check_mfa_oov.py")
    print("       -> confirm OOV word types / tokens have dropped")
    print("  2. mfa validate mfa_segments/ hindi_indic hindi_indic \\")
    print("       --num_jobs 4")
    print("       -> confirm MFA's own OOV warning has also dropped")
    print("  3. Proceed to full mfa align once confirmed")
    print("=" * 65)


if __name__ == "__main__":
    main()