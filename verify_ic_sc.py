#!/usr/bin/env python3
"""
verify_ic_sc.py — Compare Prof's IC/SC vs Your IC/SC on sample sentences

Picks 5 sample sentences from the .conllu file and shows side-by-side
comparison of IC/SC values from both implementations.

Usage:
  python verify_ic_sc.py

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import sys
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
CONLLU_PATH = r"/mnt/c/Users/MayurGaidhane/Downloads/indic_pipeline_full_dataset/indic_language_pipeline/outputs/results/features_B_stanza.conllu"
NUM_SAMPLES = 5
random.seed(42)

# ══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════
@dataclass
class Token:
    id: int
    form: str
    upos: str
    head: int
    deprel: str

@dataclass
class Sentence:
    sent_id: str
    text: str
    tokens: List[Token] = field(default_factory=list)

# ══════════════════════════════════════════════════════════════
#  CONLLU PARSER
# ══════════════════════════════════════════════════════════════
DEFAULT_EXCLUDED_UPOS = {"PUNCT"}

def parse_conllu(filepath: str) -> List[Sentence]:
    sentences = []
    current_tokens = []
    current_sent_id = None
    current_text = None
    counter = 0

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current_tokens:
                    counter += 1
                    sid = current_sent_id or f"sent_{counter}"
                    text = current_text or " ".join(t.form for t in current_tokens)
                    sentences.append(Sentence(sent_id=sid, text=text, tokens=current_tokens))
                current_tokens = []
                current_sent_id = None
                current_text = None
                continue

            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        current_sent_id = parts[1].strip()
                elif line.startswith("# text"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        current_text = parts[1].strip()
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                continue
            tok_id_raw = fields[0]
            if "-" in tok_id_raw or "." in tok_id_raw:
                continue
            try:
                tok_id = int(tok_id_raw)
                form = fields[1]
                upos = fields[3]
                head_raw = fields[6]
                if head_raw == "_":
                    continue
                head = int(head_raw)
                deprel = fields[7]
                current_tokens.append(Token(id=tok_id, form=form, upos=upos, head=head, deprel=deprel))
            except (ValueError, IndexError):
                continue

    # Flush last sentence
    if current_tokens:
        counter += 1
        sid = current_sent_id or f"sent_{counter}"
        text = current_text or " ".join(t.form for t in current_tokens)
        sentences.append(Sentence(sent_id=sid, text=text, tokens=current_tokens))

    return sentences


# ══════════════════════════════════════════════════════════════
#  PROF'S IC/SC (from dlt.py)
# ══════════════════════════════════════════════════════════════
def is_countable_word(upos: str) -> bool:
    return upos not in DEFAULT_EXCLUDED_UPOS

def prof_ic(tokens: List[Token]) -> List[int]:
    """Prof's Integration Cost — baseline 1 + intervening non-PUNCT distance,
    assigned to the LATER token in each dependency."""
    id_to_token = {t.id: t for t in tokens}
    ic = {t.id: 1 for t in tokens}

    for tok in tokens:
        dep_id = tok.id
        head_id = tok.head
        if head_id == 0 or head_id not in id_to_token:
            continue
        lower = min(dep_id, head_id)
        upper = max(dep_id, head_id)
        later_id = upper
        intervening = [t for t in tokens if lower < t.id < upper]
        length = sum(1 for t in intervening if is_countable_word(t.upos))
        ic[later_id] = ic.get(later_id, 1) + length

    # Zero out PUNCT
    for t in tokens:
        if t.upos in DEFAULT_EXCLUDED_UPOS:
            ic[t.id] = 0

    return [ic[t.id] for t in tokens]


def prof_sc(tokens: List[Token]) -> List[int]:
    """Prof's Storage Cost — baseline 1 + count of incomplete deps (d <= i < h)."""
    ids_sorted = sorted(t.id for t in tokens)
    valid_ids = set(ids_sorted)
    dependencies = [(t.id, t.head) for t in tokens if t.head != 0 and t.head in valid_ids]

    sc = {}
    for i in ids_sorted:
        unresolved = sum(1 for (d, h) in dependencies if d <= i < h)
        sc[i] = 1 + unresolved

    # Zero out PUNCT
    for t in tokens:
        if t.upos in DEFAULT_EXCLUDED_UPOS:
            sc[t.id] = 0

    return [sc[t.id] for t in tokens]


# ══════════════════════════════════════════════════════════════
#  YOUR IC/SC (from phase_B_stanza.py — FIXED version)
# ══════════════════════════════════════════════════════════════
def your_ic(tokens: List[Token]) -> List[int]:
    """Your IC — baseline 1 + intervening non-PUNCT, assigned to LATER token
    (both directions, matching Prof's logic and SCiL 2021 paper)."""
    id_to_token = {t.id: t for t in tokens}
    ic = {t.id: 1 for t in tokens}

    for tok in tokens:
        dep_id = tok.id
        head_id = tok.head
        if head_id == 0 or head_id not in id_to_token:
            continue
        lower = min(dep_id, head_id)
        upper = max(dep_id, head_id)
        intervening = sum(1 for t in tokens if lower < t.id < upper and t.upos not in DEFAULT_EXCLUDED_UPOS)
        ic[upper] = ic.get(upper, 1) + intervening

    # Zero out PUNCT
    for t in tokens:
        if t.upos in DEFAULT_EXCLUDED_UPOS:
            ic[t.id] = 0

    return [ic[t.id] for t in tokens]


def your_sc(tokens: List[Token]) -> List[int]:
    """Your SC — baseline 1 + count of incomplete deps (d <= i < h),
    matching Prof's logic."""
    ids_sorted = sorted(t.id for t in tokens)
    valid_ids = set(ids_sorted)
    dependencies = [(t.id, t.head) for t in tokens if t.head != 0 and t.head in valid_ids]

    sc = {}
    for i in ids_sorted:
        unresolved = sum(1 for (d, h) in dependencies if d <= i < h)
        sc[i] = 1 + unresolved

    # Zero out PUNCT
    for t in tokens:
        if t.upos in DEFAULT_EXCLUDED_UPOS:
            sc[t.id] = 0

    return [sc[t.id] for t in tokens]


# ══════════════════════════════════════════════════════════════
#  MAIN — COMPARE
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  IC/SC VERIFICATION: Prof's Code vs Your Code")
    print("  Mayur Shriram Gaidhane (M22AIE248)")
    print("=" * 70)

    print(f"\n  Loading {CONLLU_PATH} ...")
    sentences = parse_conllu(CONLLU_PATH)
    print(f"  Total sentences parsed: {len(sentences):,}")

    # Filter: pick sentences with 4-10 tokens (easy to read)
    good_sents = [s for s in sentences if 4 <= len(s.tokens) <= 10]
    print(f"  Sentences with 4-10 tokens: {len(good_sents):,}")

    if len(good_sents) < NUM_SAMPLES:
        samples = good_sents
    else:
        samples = random.sample(good_sents, NUM_SAMPLES)

    print(f"\n  Comparing {len(samples)} sample sentences ...\n")

    total_ic_match = 0
    total_ic_diff = 0
    total_sc_match = 0
    total_sc_diff = 0

    for idx, sent in enumerate(samples, 1):
        tokens = sent.tokens
        words = [t.form for t in tokens]
        upos_list = [t.upos for t in tokens]

        # Compute both
        p_ic = prof_ic(tokens)
        p_sc = prof_sc(tokens)
        y_ic = your_ic(tokens)
        y_sc = your_sc(tokens)

        # Check matches
        ic_match = (p_ic == y_ic)
        sc_match = (p_sc == y_sc)

        print("=" * 70)
        print(f"  SAMPLE {idx}: {sent.sent_id}")
        print(f"  Text: {sent.text[:80]}")
        print("-" * 70)
        print(f"  {'Token':<15} {'POS':<6} {'Head':<5} "
              f"{'Prof_IC':>8} {'Your_IC':>8} {'IC_OK':>6} "
              f"{'Prof_SC':>8} {'Your_SC':>8} {'SC_OK':>6}")
        print(f"  {'-'*68}")

        for j, t in enumerate(tokens):
            ic_ok = "✓" if p_ic[j] == y_ic[j] else "✗"
            sc_ok = "✓" if p_sc[j] == y_sc[j] else "✗"
            print(f"  {t.form:<15} {t.upos:<6} {t.head:<5} "
                  f"{p_ic[j]:>8} {y_ic[j]:>8} {ic_ok:>6} "
                  f"{p_sc[j]:>8} {y_sc[j]:>8} {sc_ok:>6}")

            if p_ic[j] == y_ic[j]:
                total_ic_match += 1
            else:
                total_ic_diff += 1
            if p_sc[j] == y_sc[j]:
                total_sc_match += 1
            else:
                total_sc_diff += 1

        print()
        print(f"  IC match: {'YES ✓' if ic_match else 'NO ✗ — DIFFERS'}")
        print(f"  SC match: {'YES ✓' if sc_match else 'NO ✗ — DIFFERS'}")
        print(f"  Prof IC: {p_ic}")
        print(f"  Your IC: {y_ic}")
        print(f"  Prof SC: {p_sc}")
        print(f"  Your SC: {y_sc}")
        print()

    # Summary
    total_tokens = total_ic_match + total_ic_diff
    print("=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Sentences compared : {len(samples)}")
    print(f"  Total tokens       : {total_tokens}")
    print()
    print(f"  IC matches         : {total_ic_match}/{total_tokens} "
          f"({100*total_ic_match/total_tokens:.1f}%)")
    print(f"  IC differences     : {total_ic_diff}/{total_tokens}")
    print()
    print(f"  SC matches         : {total_sc_match}/{total_tokens} "
          f"({100*total_sc_match/total_tokens:.1f}%)")
    print(f"  SC differences     : {total_sc_diff}/{total_tokens}")
    print()

    if total_ic_diff == 0 and total_sc_diff == 0:
        print("  RESULT: ✓ PERFECT MATCH — Your IC/SC matches Prof's code")
    else:
        print("  RESULT: ✗ DIFFERENCES FOUND — IC/SC computation needs fixing")
        if total_ic_diff > 0:
            print("    → IC differs: Check both-direction assignment and PUNCT exclusion")
        if total_sc_diff > 0:
            print("    → SC differs: Check boundary conditions (d<=i<h)")
    print("=" * 70)


if __name__ == "__main__":
    main()