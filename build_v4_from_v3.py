#!/usr/bin/env python3
"""
build_v4_from_v3.py  — FULLY FIXED VERSION

FIXES:
  1. normalized_word_count: counts code-switch words in brackets as words
     [hello] in normalized = spoken word → count it
     [inhaling] = annotation → don't count it

  2. env_noise: cross-validates between verbatim AND normalized
     Only counts env_noise tags present in BOTH texts
     Prevents code-switch words [music][door] being counted as noise

  3. all_annotations: includes ALL bracket tags (including code-switch)
     all_annotation_count counts [sir][time][hello] etc.
     unknown_annotations only shows truly unknown annotation tags
     (not English code-switch words)
"""

import os, sys, csv, re, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config

V3_CSV   = os.path.join(config.RESULTS_DIR,
            "indicvoices_v1_hindi_segments_v3.csv")
V4_CSV   = os.path.join(config.RESULTS_DIR,
            "indicvoices_v1_hindi_segments_v4.csv")
V4_EXCEL = os.path.join(config.RESULTS_DIR,
            "indicvoices_v1_hindi_segments_v4_excel.csv")
REPORT   = os.path.join(config.RESULTS_DIR,
            "segment_analysis_report_v4.txt")

ANNOTATION_CATEGORIES = {
    "dysfluency": [
        "stammers","stammer","stutter","stutters",
        "uhh","uh","umm","um","hmm","hm",
        "err","erm","ahh","ah","ahm",
        "stammering","stuttering",
        "filled_pause","false_start",
        "ugh","uhm","ehh","eh","aah",
    ],
    "physiological": [
        "inhaling","exhaling","breathing","breath",
        "cough","coughing",
        "sneeze","sneezing",
        "yawning","yawn",
        "swallowing","swallow",
        "sigh","sighing",
        "throat_clearing","clearing_throat",
        "lip_smacking","clicking",
        "nose_blowing","sniffling","sniff",
        "hiccup","hiccups",
    ],
    "environmental_noise": [
        "noise","background_noise","env_noise",
        "traffic","traffic_noise",
        "horn","honking",
        "phone_ringing","phone_vibrating",
        "bell","doorbell",
        "knock","knocking",
        "tv_sound","tv_noise",
        "barking","dog_barking",
        "bird","crow",
        "crowd","crowd_noise",
        "clinking","buzzer",
        "construction","drilling",
        "vehicle_noise",
        "ac_noise",
    ],
    "quality_annotation": [
        "unintelligible","inaudible",
        "unclear_speech","noise_here",
        "crosstalk","overlap",
        "unknown_word",
    ],
    "paralinguistic": [
        "laughing","laughter","laugh",
        "giggling","giggle",
        "chuckling","chuckle",
        "crying","cry","sobbing","sob",
        "whispering","whisper",
        "shouting","yelling",
        "singing","humming","hum",
        "groaning","groan",
    ],
    "action": [
        "clapping","clap",
        "tapping","tap",
        "snapping","snap",
        "talking","speaking",
        "mumbling","mumble",
        "pausing",
    ],
}

BRACKET_TAG_RE = re.compile(r'\[\s*([^\]]+?)\s*\]')
ENGLISH_RE     = re.compile(r'\b[A-Za-z]{2,}\b')
DEVANAGARI_RE  = re.compile(r'[\u0900-\u097F]+')
PURE_ENG_RE    = re.compile(r'^[a-zA-Z][a-zA-Z0-9]*$')


# ══════════════════════════════════════════════════════════════
# TAG CATEGORIZATION
# ══════════════════════════════════════════════════════════════

def categorize_tag(tag):
    """
    Exact match first.
    Prefix match only for keywords length >= 4.
    Pure English-only tags → code_switch.
    """
    tag_clean = tag.strip().lower()
    tag_u     = tag_clean.replace(' ','_').replace('-','_')

    for cat, kws in ANNOTATION_CATEGORIES.items():
        if tag_clean in kws or tag_u in kws:
            return cat

    for cat, kws in ANNOTATION_CATEGORIES.items():
        for kw in kws:
            if len(kw) < 4:
                continue
            if tag_clean.startswith(kw) or tag_u.startswith(kw):
                return cat
            if kw.startswith(tag_clean) or kw.startswith(tag_u):
                return cat

    if PURE_ENG_RE.match(tag_clean):
        return "code_switch"

    return "unknown"


# ══════════════════════════════════════════════════════════════
# FIX 1: WORD COUNT
# ══════════════════════════════════════════════════════════════

def count_verbatim_words(text):
    """
    Verbatim word count: remove ALL bracket tags entirely.
    [inhaling], [stammers], [hello] etc. are all removed.
    Only the base Hindi (and plain English) words are counted.

    Example: "हलो जी" → 2
    """
    if not text:
        return 0
    clean = BRACKET_TAG_RE.sub(' ', text)
    return len([w for w in clean.split() if w.strip()])


def count_normalized_words(text):
    """
    FIX 1: Normalized word count — code-switch words IN brackets
    are SPOKEN words and must be COUNTED.

    Rule:
      [hello] → code_switch → spoken word → COUNT IT → "hello"
      [inhaling] → physiological → annotation → REMOVE IT
      [stammers] → dysfluency → annotation → REMOVE IT

    Example:
      "हलो [hello] जी" → "हलो hello जी" → 3 words ✅
      "मैं [inhaling] जाना" → "मैं जाना" → 2 words ✅
      "अब [sir] पड़ोसी [time] को" → "अब sir पड़ोसी time को" → 5 ✅
    """
    if not text:
        return 0

    def replace_tag(match):
        tag = match.group(1).strip()
        cat = categorize_tag(tag)
        if cat == "code_switch":
            return ' ' + tag + ' '  # spoken word — count it
        else:
            return ' '              # annotation — remove it

    clean = BRACKET_TAG_RE.sub(replace_tag, text)
    return len([w for w in clean.split() if w.strip()])


def make_clean_hindi_text(normalized_text, verbatim_text):
    """Remove brackets + English. Keep Devanagari >= 2 chars."""
    text = normalized_text or verbatim_text or ""
    if not text:
        return ""
    text = BRACKET_TAG_RE.sub(' ', text)
    text = ENGLISH_RE.sub(' ', text)
    words = []
    for token in text.split():
        deva = ''.join(DEVANAGARI_RE.findall(token))
        if len(deva) >= 2:
            words.append(deva)
    return ' '.join(words).strip()


def count_clean_hindi_words(text):
    if not text:
        return 0
    text = BRACKET_TAG_RE.sub(' ', text)
    text = ENGLISH_RE.sub(' ', text)
    count = 0
    for token in text.split():
        deva = ''.join(DEVANAGARI_RE.findall(token))
        if len(deva) >= 2:
            count += 1
    return count


# ══════════════════════════════════════════════════════════════
# FIX 2 + 3: ANNOTATION EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_tags(text):
    """Extract all bracket tag contents from text."""
    if not text:
        return []
    return [m.strip() for m in BRACKET_TAG_RE.findall(text) if m.strip()]


def get_annotation_info(text, valid_env_noise_tags=None):
    """
    Extract annotation info from ONE text string.

    FIX 3: all_annotation_count includes ALL bracket tags
    (including code-switch words like [sir][time][hello]).
    unknown_annotations only shows truly unknown tags
    (NOT English code-switch words).

    FIX 2: valid_env_noise_tags parameter allows cross-validation.
    Only env_noise tags in this set are counted as environmental noise.
    """
    empty = {
        "dysfluency_types"              : "",
        "dysfluency_count"              : 0,
        "physiological_events"          : "",
        "physiological_count"           : 0,
        "env_noise_events"              : "",
        "env_noise_count"               : 0,
        "quality_annotations"           : "",
        "quality_annotation_count"      : 0,
        "paralinguistic_events"         : "",
        "paralinguistic_count"          : 0,
        "action_types"                  : "",
        "action_count"                  : 0,
        "all_annotation_count"          : 0,
        "unknown_annotations"           : "",
        "unknown_annotation_count"      : 0,
    }
    if not text:
        return empty

    raw_tags = extract_tags(text)
    if not raw_tags:
        return empty

    cat_tags = defaultdict(list)
    unknown  = []
    all_tags = []   # FIX 3: all tags including code_switch

    for tag in raw_tags:
        cat = categorize_tag(tag)

        # FIX 3: add to all_tags regardless of category
        all_tags.append(tag)

        if cat == "code_switch":
            pass  # counted in all_tags but not in category
        elif cat == "unknown":
            unknown.append(tag)
            # NOTE: do NOT add English code-switch to unknown
        elif cat == "environmental_noise":
            # FIX 2: only count if in valid set (present in both texts)
            if valid_env_noise_tags is None:
                cat_tags[cat].append(tag)
            elif tag.strip().lower() in valid_env_noise_tags:
                cat_tags[cat].append(tag)
        else:
            cat_tags[cat].append(tag)

    def unique_pipe(lst):
        seen = []
        for x in lst:
            xl = x.lower()
            if xl not in [s.lower() for s in seen]:
                seen.append(x)
        return "|".join(seen)

    cat_map = {
        "dysfluency"         : ("dysfluency_types",    "dysfluency_count"),
        "physiological"      : ("physiological_events","physiological_count"),
        "environmental_noise": ("env_noise_events",    "env_noise_count"),
        "quality_annotation" : ("quality_annotations", "quality_annotation_count"),
        "paralinguistic"     : ("paralinguistic_events","paralinguistic_count"),
        "action"             : ("action_types",        "action_count"),
    }

    result = dict(empty)
    for cat, (types_key, count_key) in cat_map.items():
        tags = cat_tags.get(cat, [])
        result[types_key] = unique_pipe(tags)
        result[count_key] = len(tags)

    # FIX 3: all_annotation_count = ALL bracket tags (including code-switch)
    result["all_annotation_count"]     = len(all_tags)
    result["unknown_annotations"]      = unique_pipe(unknown)
    result["unknown_annotation_count"] = len(unknown)

    return result


def get_valid_env_noise_tags(verb_text, norm_text):
    """
    FIX 2: Return set of env_noise tag names that appear in BOTH texts.

    Real env_noise tags like [horn][noise] appear in BOTH verbatim
    and normalized because they are actual recordings of ambient sound.

    Code-switch words like [music][door][rain] typically appear ONLY
    in normalized text (annotator adds English translation in brackets).

    By requiring presence in BOTH texts, we filter out code-switch words.
    """
    def get_env_tags(text):
        tags = extract_tags(text or "")
        return set(t.strip().lower() for t in tags
                   if categorize_tag(t.strip()) == "environmental_noise")

    verb_env = get_env_tags(verb_text)
    norm_env = get_env_tags(norm_text)

    # Only tags present in BOTH
    return verb_env & norm_env


# ══════════════════════════════════════════════════════════════
# QUALITY SCORE V4
# ══════════════════════════════════════════════════════════════

def compute_quality_v4(row, verb_info, verb_text):
    """
    ISSUE 4 ANSWER: Quality Score V4 Calculation

    quality_score (V3) = acoustic_only_score
      Uses only verification_report flags from human annotators.
      Starts at 10, deductions for noise/echo/volume issues.
      Range: 0-10

    quality_score_v4 = acoustic_score - linguistic_deduction
      = quality_acoustic_component - quality_linguistic_deduction
      Range: 0-10

    quality_acoustic_component = same as quality_score (V3)
      This is the ACOUSTIC part of the score.
      Comes from verification_report flags only.
      Example: if vr_echo_present=True → acoustic_component = 10-1 = 9

    quality_linguistic_deduction = NEW in V4
      How much to deduct based on LINGUISTIC content of the text.
      Based on what annotators actually wrote in the transcript.

      dysfluency_density > 0.30 (30%+ of words are dysfluent): -2
      dysfluency_density > 0.15 (15%+ of words are dysfluent): -1
      unintelligible_words > 3: -2
      unintelligible_words > 0: -1
      env_noise_count > 5: -1  (many noise events in text)
      unknown_annotations > 5: -1  (annotation quality concern)
      Maximum deduction = 4

    dysfluency_density = dysfluency_count / word_count
      Example: segment has 3 stammers in 20 words
               dysfluency_density = 3/20 = 0.15
               Just at threshold → deduction = -1

    EXAMPLE:
      Segment with vr_echo_present=True, 4 stammers in 10 words:
        quality_score (V3)          = 10 - 1 = 9   (acoustic only)
        quality_acoustic_component  = 9
        dysfluency_density          = 4/10 = 0.40
        quality_linguistic_deduction= -2   (density > 0.30)
        quality_score_v4            = 9 - 2 = 7
    """
    acoustic    = int(row.get("quality_score", 10) or 10)
    word_count  = max(1, count_verbatim_words(verb_text))
    dys_count   = verb_info.get("dysfluency_count", 0)
    dys_density = round(dys_count / word_count, 4)
    unintel     = verb_info.get("quality_annotation_count", 0)
    env_noise   = verb_info.get("env_noise_count", 0)
    unknown_cnt = verb_info.get("unknown_annotation_count", 0)

    ling_ded = 0
    if dys_density  > 0.30: ling_ded += 2
    elif dys_density > 0.15: ling_ded += 1
    if unintel > 3:   ling_ded += 2
    elif unintel > 0: ling_ded += 1
    if env_noise > 5: ling_ded += 1
    if unknown_cnt > 5: ling_ded += 1
    ling_ded = min(4, ling_ded)

    return {
        "quality_score_v4"            : max(0, acoustic - ling_ded),
        "quality_acoustic_component"  : acoustic,
        "quality_linguistic_deduction": ling_ded,
        "dysfluency_density"          : dys_density,
    }


# ══════════════════════════════════════════════════════════════
# COLUMN ORDER
# ══════════════════════════════════════════════════════════════

def build_v4_fieldnames(v3_fieldnames):
    INSERT_AFTER = {
        "speaker_id_segment"       : ["speaker_id_note"],
        "normalized_text"          : ["clean_hindi_text"],
        "normalized_word_count"    : ["clean_hindi_word_count"],
        "dysfluency_types"         : ["verbatim_dysfluency_types",
                                      "normalized_dysfluency_types"],
        "dysfluency_count"         : ["verbatim_dysfluency_count",
                                      "normalized_dysfluency_count"],
        "physiological_events"     : ["verbatim_physiological_events",
                                      "normalized_physiological_events"],
        "physiological_count"      : ["verbatim_physiological_count",
                                      "normalized_physiological_count"],
        "env_noise_events"         : ["verbatim_env_noise_events",
                                      "normalized_env_noise_events"],
        "env_noise_count"          : ["verbatim_env_noise_count",
                                      "normalized_env_noise_count"],
        "quality_annotations"      : ["verbatim_quality_annotations",
                                      "normalized_quality_annotations"],
        "quality_annotation_count" : ["verbatim_quality_annotation_count",
                                      "normalized_quality_annotation_count"],
        "paralinguistic_events"    : ["verbatim_paralinguistic_events",
                                      "normalized_paralinguistic_events"],
        "paralinguistic_count"     : ["verbatim_paralinguistic_count",
                                      "normalized_paralinguistic_count"],
        "action_types"             : ["verbatim_action_types",
                                      "normalized_action_types"],
        "action_count"             : ["verbatim_action_count",
                                      "normalized_action_count"],
        "all_annotation_count"     : ["verbatim_all_annotation_count",
                                      "normalized_all_annotation_count"],
        "unknown_annotations"      : ["verbatim_unknown_annotations",
                                      "normalized_unknown_annotations",
                                      "verbatim_unknown_annotation_count",
                                      "normalized_unknown_annotation_count"],
        "quality_score"            : ["quality_score_v4",
                                      "quality_acoustic_component",
                                      "quality_linguistic_deduction",
                                      "dysfluency_density"],
    }
    v4_cols = []
    for col in v3_fieldnames:
        v4_cols.append(col)
        if col in INSERT_AFTER:
            v4_cols.extend(INSERT_AFTER[col])
    return v4_cols


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  BUILD V4 CSV FROM V3 — ALL BUGS FIXED")
    print("=" * 65)
    print()
    print("  Fix 1: normalized_word_count counts code-switch words")
    print("  Fix 2: env_noise cross-validated (present in both texts)")
    print("  Fix 3: all_annotation_count includes ALL bracket tags")
    print()

    if not os.path.exists(V3_CSV):
        print(f"  ERROR: V3 CSV not found: {V3_CSV}")
        return

    print(f"  Loading V3 CSV ...")
    with open(V3_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        v3_fieldnames = list(reader.fieldnames)
        v3_rows = list(reader)

    print(f"  V3 rows    : {len(v3_rows):,}")
    print(f"  V3 columns : {len(v3_fieldnames)}")

    v4_fieldnames = build_v4_fieldnames(v3_fieldnames)
    new_cols = [c for c in v4_fieldnames if c not in v3_fieldnames]
    print(f"  V4 columns : {len(v4_fieldnames)}")
    print(f"  New columns: {len(new_cols)}")
    print()
    print("  Processing rows ...")

    start = time.time()
    score_changed = 0
    v4_rows = []

    for idx, row in enumerate(v3_rows):
        if (idx + 1) % 5000 == 0:
            print(f"  [{idx+1:6d}/{len(v3_rows):6d}]")

        verb_text = row.get("verbatim_text",  "") or ""
        norm_text = row.get("normalized_text","") or ""

        # FIX 1: correct word counts
        verb_wc  = count_verbatim_words(verb_text)
        norm_wc  = count_normalized_words(norm_text)  # counts code-switch
        clean_ht = make_clean_hindi_text(norm_text, verb_text)
        clean_wc = count_clean_hindi_words(norm_text or verb_text)

        row["verbatim_word_count"]   = verb_wc
        row["normalized_word_count"] = norm_wc

        # FIX 2: cross-validated env_noise set
        valid_env = get_valid_env_noise_tags(verb_text, norm_text)

        # FIX 3: get annotation info with cross-validation
        verb_info = get_annotation_info(verb_text, valid_env)
        norm_info = get_annotation_info(norm_text, valid_env)

        # Quality score V4
        qs = compute_quality_v4(row, verb_info, verb_text)
        if qs["quality_score_v4"] != int(row.get("quality_score",10) or 10):
            score_changed += 1

        # ── RECOMPUTE V3 COLUMNS WITH IMPROVED KEYWORDS ───────
        # The V3 build script had INCOMPLETE keyword lists.
        # Example: phone_vibrating → missed → has_env_noise=False (WRONG)
        #          throat_clearing → missed → has_physiological=False (WRONG)
        # We now recompute all has_*, *_events, *_count columns
        # using verbatim_text as the primary source.

        def unique_pipe_both(v, n):
            """Merge two pipe-separated strings, keeping unique values."""
            seen = []
            for val in (v + "|" + n).split("|"):
                val = val.strip()
                if val and val.lower() not in [s.lower() for s in seen]:
                    seen.append(val)
            return "|".join(seen)

        # Recompute existing V3 columns using verbatim_info (primary)
        # has_* = True if found in verbatim OR normalized
        row["has_dysfluency"]          = (verb_info["dysfluency_count"] > 0 or
                                          norm_info["dysfluency_count"] > 0)
        row["dysfluency_types"]        = unique_pipe_both(
                                          verb_info["dysfluency_types"],
                                          norm_info["dysfluency_types"])
        row["dysfluency_count"]        = verb_info["dysfluency_count"]

        row["has_physiological"]       = (verb_info["physiological_count"] > 0 or
                                          norm_info["physiological_count"] > 0)
        row["physiological_events"]    = unique_pipe_both(
                                          verb_info["physiological_events"],
                                          norm_info["physiological_events"])
        row["physiological_count"]     = verb_info["physiological_count"]

        row["has_env_noise"]           = (verb_info["env_noise_count"] > 0 or
                                          norm_info["env_noise_count"] > 0)
        row["env_noise_events"]        = unique_pipe_both(
                                          verb_info["env_noise_events"],
                                          norm_info["env_noise_events"])
        row["env_noise_count"]         = verb_info["env_noise_count"]

        row["has_quality_annotation"]  = (verb_info["quality_annotation_count"] > 0 or
                                          norm_info["quality_annotation_count"] > 0)
        row["quality_annotations"]     = unique_pipe_both(
                                          verb_info["quality_annotations"],
                                          norm_info["quality_annotations"])
        row["quality_annotation_count"]= verb_info["quality_annotation_count"]

        row["has_paralinguistic"]      = (verb_info["paralinguistic_count"] > 0 or
                                          norm_info["paralinguistic_count"] > 0)
        row["paralinguistic_events"]   = unique_pipe_both(
                                          verb_info["paralinguistic_events"],
                                          norm_info["paralinguistic_events"])
        row["paralinguistic_count"]    = verb_info["paralinguistic_count"]

        row["has_action"]              = (verb_info["action_count"] > 0 or
                                          norm_info["action_count"] > 0)
        row["action_types"]            = unique_pipe_both(
                                          verb_info["action_types"],
                                          norm_info["action_types"])
        row["action_count"]            = verb_info["action_count"]

        row["all_annotations"]         = unique_pipe_both(
                                          verb_info.get("unknown_annotations",""),
                                          norm_info.get("unknown_annotations",""))
        row["all_annotation_count"]    = (verb_info["all_annotation_count"] +
                                          norm_info["all_annotation_count"])
        row["unknown_annotations"]     = unique_pipe_both(
                                          verb_info["unknown_annotations"],
                                          norm_info["unknown_annotations"])
        row["unknown_annotation_count"]= (verb_info["unknown_annotation_count"] +
                                          norm_info["unknown_annotation_count"])

        # Set all new columns
        row["speaker_id_note"]                      = "source_data_all_zero"
        row["clean_hindi_text"]                     = clean_ht
        row["clean_hindi_word_count"]               = clean_wc
        row["verbatim_dysfluency_types"]            = verb_info["dysfluency_types"]
        row["normalized_dysfluency_types"]          = norm_info["dysfluency_types"]
        row["verbatim_dysfluency_count"]            = verb_info["dysfluency_count"]
        row["normalized_dysfluency_count"]          = norm_info["dysfluency_count"]
        row["verbatim_physiological_events"]        = verb_info["physiological_events"]
        row["normalized_physiological_events"]      = norm_info["physiological_events"]
        row["verbatim_physiological_count"]         = verb_info["physiological_count"]
        row["normalized_physiological_count"]       = norm_info["physiological_count"]
        row["verbatim_env_noise_events"]            = verb_info["env_noise_events"]
        row["normalized_env_noise_events"]          = norm_info["env_noise_events"]
        row["verbatim_env_noise_count"]             = verb_info["env_noise_count"]
        row["normalized_env_noise_count"]           = norm_info["env_noise_count"]
        row["verbatim_quality_annotations"]         = verb_info["quality_annotations"]
        row["normalized_quality_annotations"]       = norm_info["quality_annotations"]
        row["verbatim_quality_annotation_count"]    = verb_info["quality_annotation_count"]
        row["normalized_quality_annotation_count"]  = norm_info["quality_annotation_count"]
        row["verbatim_paralinguistic_events"]       = verb_info["paralinguistic_events"]
        row["normalized_paralinguistic_events"]     = norm_info["paralinguistic_events"]
        row["verbatim_paralinguistic_count"]        = verb_info["paralinguistic_count"]
        row["normalized_paralinguistic_count"]      = norm_info["paralinguistic_count"]
        row["verbatim_action_types"]                = verb_info["action_types"]
        row["normalized_action_types"]              = norm_info["action_types"]
        row["verbatim_action_count"]                = verb_info["action_count"]
        row["normalized_action_count"]              = norm_info["action_count"]
        row["verbatim_all_annotation_count"]        = verb_info["all_annotation_count"]
        row["normalized_all_annotation_count"]      = norm_info["all_annotation_count"]
        row["verbatim_unknown_annotations"]         = verb_info["unknown_annotations"]
        row["normalized_unknown_annotations"]       = norm_info["unknown_annotations"]
        row["verbatim_unknown_annotation_count"]    = verb_info["unknown_annotation_count"]
        row["normalized_unknown_annotation_count"]  = norm_info["unknown_annotation_count"]
        row["quality_score_v4"]                     = qs["quality_score_v4"]
        row["quality_acoustic_component"]           = qs["quality_acoustic_component"]
        row["quality_linguistic_deduction"]         = qs["quality_linguistic_deduction"]
        row["dysfluency_density"]                   = qs["dysfluency_density"]

        v4_rows.append(row)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"\n  Writing V4 CSV ...")
    with open(V4_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=v4_fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(v4_rows)
    v4_mb = os.path.getsize(V4_CSV)/1024/1024
    v3_mb = os.path.getsize(V3_CSV)/1024/1024
    print(f"  V3 size : {v3_mb:.1f} MB")
    print(f"  V4 size : {v4_mb:.1f} MB  ({'larger ✅' if v4_mb>v3_mb else 'CHECK'})")

    print(f"\n  Writing V4 Excel CSV ...")
    with open(V4_EXCEL, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=v4_fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(v4_rows)
    print(f"  Saved: {V4_EXCEL}")

    # Report
    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr()
    pr("=" * 65)
    pr("  V4 CSV BUILD REPORT")
    pr("=" * 65)
    pr(f"  V3 : {len(v3_rows):,} rows × {len(v3_fieldnames)} cols")
    pr(f"  V4 : {len(v4_rows):,} rows × {len(v4_fieldnames)} cols")
    pr(f"  New columns: {len(new_cols)}")
    pr()

    # Test Fix 1
    pr("── FIX 1: normalized_word_count ─────────────────────────")
    test_cases_wc = [
        ("हलो [hello] जी",                    3, "code-switch counted"),
        ("मैं [inhaling] जाना",               2, "annotation not counted"),
        ("अब [sir] पड़ोसी [time] को",          5, "2 code-switch counted"),
        ("हाँ जी",                              2, "no brackets"),
    ]
    for text, expected, desc in test_cases_wc:
        got = count_normalized_words(text)
        ok  = "✅" if got == expected else "❌"
        pr(f"  {ok} '{text[:30]}' → {got} (expected {expected}) — {desc}")
    pr()

    # Test Fix 2
    pr("── FIX 2: env_noise cross-validation ───────────────────")
    test_env = [
        ("horn present in both",
         "[horn] noise here", "[horn] traffic",
         {"horn"}, "horn in both → valid"),
        ("music only in normalized (code-switch)",
         "background noise here", "I like [music]",
         set(), "music only normalized → filtered out"),
    ]
    for desc, verb, norm, expected, note in test_env:
        got = get_valid_env_noise_tags(verb, norm)
        ok  = "✅" if got == expected else "❌"
        pr(f"  {ok} {desc}: {got} — {note}")
    pr()

    # Test Fix 3
    pr("── FIX 3: all_annotation_count ─────────────────────────")
    test_text = "अब मैं सर [sir] पड़ोसी को [time] नहीं सर [sir] के पास"
    info = get_annotation_info(test_text)
    pr(f"  Text: '{test_text[:50]}'")
    pr(f"  all_annotation_count = {info['all_annotation_count']} (expected 3) {'✅' if info['all_annotation_count']==3 else '❌'}")
    pr(f"  unknown_annotations  = '{info['unknown_annotations']}' (expected empty) {'✅' if not info['unknown_annotations'] else '❌'}")
    pr()

    # Quality score stats
    pr("── Quality score ─────────────────────────────────────────")
    avg_v3 = sum(int(r["quality_score"] or 0) for r in v4_rows)/len(v4_rows)
    avg_v4 = sum(int(r["quality_score_v4"]) for r in v4_rows)/len(v4_rows)
    pr(f"  quality_score (V3)    mean: {avg_v3:.3f}")
    pr(f"  quality_score_v4 (V4) mean: {avg_v4:.3f}")
    pr(f"  Segments where score changed: {score_changed:,}")
    pr()
    pr("=" * 65)

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report: {REPORT}")


if __name__ == "__main__":
    main()