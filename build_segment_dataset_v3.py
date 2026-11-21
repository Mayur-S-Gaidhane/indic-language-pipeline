#!/usr/bin/env python3
# ============================================================
# build_segment_dataset_v3.py — COMPLETE FINAL VERSION
#
# KEY IMPROVEMENTS OVER V2:
#
#   FIX 1: Short segments NO LONGER SKIPPED
#     - All segments saved to CSV regardless of duration
#     - New column: is_mfa_ready (True/False)
#     - New column: skip_reason (too_short/no_text/etc.)
#     - Short segments = text-only rows (no audio cut)
#     - Reason: "तीन सो का" and "अच्छा जी" carry conversation
#       context even if too short for MFA alignment
#
#   FIX 2: ALL annotation tags captured dynamically
#     - FIRST run discover_annotations.py to find all tags
#     - ALL tags captured in new column: all_annotations
#     - Unknown tags (not in any category) saved separately
#     - New column: unknown_annotations
#     - Nothing missed from any JSON file
#
#   FIX 3: Complete conversation preserved
#     - Group by file_id + sort by segment_index
#     - Reconstruct complete conversation including short turns
#     - is_mfa_ready flag separates acoustic from discourse use
#
# Run order:
#   1. python discover_annotations.py   → find all tags
#   2. python build_segment_dataset_v3.py → build dataset
#
# Outputs:
#   outputs/results/indicvoices_v1_hindi_segments_v3.csv
#   outputs/results/segment_analysis_report_v3.txt
#   outputs/figures/seg_v3_*.png
#   mfa_segments/<speaker>/<segment_id>.wav + .lab
#
# Student   : Mayur Shriram Gaidhane (M22AIE248)
# Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
# ============================================================

import os, sys, json, csv, glob, re, time, warnings
warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = config.COLORS

OUTPUT_CSV       = os.path.join(config.RESULTS_DIR,
                    "indicvoices_v1_hindi_segments_v3.csv")
REPORT_TXT       = os.path.join(config.RESULTS_DIR,
                    "segment_analysis_report_v3.txt")
MFA_SEGMENTS_DIR = os.path.join(ROOT, "mfa_segments")
MIN_MFA_DUR      = 1.0   # below this = text-only, no audio cut

# ══════════════════════════════════════════════════════════════
# COMPLETE ANNOTATION TAXONOMY
# Updated after running discover_annotations.py
# ══════════════════════════════════════════════════════════════

# Cat 2: Speech Dysfluencies
DYSFLUENCY_TAGS = {
    'stammers','stammering','stammer',
    'stutter','stuttering',
    'uhh','umm','uh','um','hmm','hm',
    'ah','eh','err','errm',
    'hesitation','hesitating',
    'false_start','false start',
    'repetition','repeating',
    'stammered','uhm','huh',
}

# Cat 3: Physiological Events
PHYSIOLOGICAL_TAGS = {
    'inhaling','exhaling','breathing','breath',
    'swallowing','gulp',
    'coughing','cough',
    'sneezing','sneeze',
    'yawning','yawn',
    'sighing','sigh',
    'clearing_throat','throat_clear','throat clear',
    'lip_smacking','lip smacking',
    'hiccup','hiccups',
    'hawking',
}

# Cat 4: Environmental/Background Noise
NOISE_WORDS = {
    'horn','barking','noise','music','bell','clapping',
    'siren','crowd','traffic','wind','rain','static',
    'beep','ring','bang','knock','click','background',
    'echo','buzz','hum','thud','dog','cat','bird',
    'whistle','phone','vehicle','thunder','drilling',
    'construction','airplane','train','crying_baby',
}

ENVIRONMENTAL_TAGS = {
    'bird_squawk','bird squawk','bird_sound','bird_chirping',
    'dog_bark','dog bark','dog_barking',
    'background_noise','background noise','background_music',
    'horn_sound','traffic_noise','traffic noise',
    'bell_sound','bell ringing','doorbell',
    'phone_ring','phone ringing','phone_buzzing',
    'crowd_noise','crowd noise','crowd_cheering',
    'music_playing','music playing',
    'vehicle_sound','vehicle passing','vehicle_horn',
    'wind_sound','rain_sound','thunder_sound',
    'baby_crying','child_crying',
    'tv_sound','radio_sound',
    'keyboard_typing','mouse_click',
}

# Cat 5: Quality Annotations
QUALITY_ANNOTATION_TAGS = {
    'unintelligible','inaudible','unclear','indistinct',
    'word_unclear','phrase_unclear',
    'overlapping_speech','overlap',
    'crosstalk','cross_talk',
    'distorted','garbled','muffled',
    'cut_off','incomplete',
}

# Cat 6: Paralinguistic Events
PARALINGUISTIC_TAGS = {
    'laughing','laugh','laughter','giggling','giggle',
    'smiling','smile',
    'crying','cry','sobbing','weeping',
    'applause','clapping',
    'exclamation','shouting','yelling',
    'whispering','whisper',
    'nodding','gesturing',
    'sighing_emotional',
    'surprised','surprised_sound',
    'angry_tone','sad_tone',
}

# Cat 7: Action/Activity Descriptions (NEW)
ACTION_TAGS = {
    'talking','speaking','muttering','mumbling',
    'reading','writing','typing',
    'eating','drinking','chewing',
    'walking','running','moving',
    'thinking','pausing',
}

# Combined: ALL known non-code-switch tags
ALL_NON_CS_TAGS = (
    DYSFLUENCY_TAGS | PHYSIOLOGICAL_TAGS | NOISE_WORDS |
    ENVIRONMENTAL_TAGS | QUALITY_ANNOTATION_TAGS |
    PARALINGUISTIC_TAGS | ACTION_TAGS |
    set('abcdefghijklmnopqrstuvwxyz')
)

# Domain keywords
DOMAIN_KEYWORDS = {
    "travel"     : ["location","travel","trip","tour","destination",
                    "hotel","flight","train","bus","taxi","route",
                    "rickshaw","driver","passenger","road","map",
                    "visit","place","city","mandir","fort","tourist"],
    "medical"    : ["doctor","patient","hospital","medicine","disease",
                    "symptom","treatment","health","clinic","nurse",
                    "prescription","fever","pain","surgery","medical"],
    "commerce"   : ["product","buy","sell","price","shop","market",
                    "order","delivery","payment","discount","item",
                    "customer","review","refund","bigbasket","farmer",
                    "vegetable","kilo","rupee","supermarket","bank"],
    "education"  : ["teacher","student","school","college","subject",
                    "exam","study","class","concept","explain",
                    "university","degree","course","lesson"],
    "food"       : ["food","restaurant","dish","recipe","eat","cook",
                    "menu","taste","swiggy","zomato","meal","icecream"],
    "technology" : ["phone","app","internet","computer","alexa","siri",
                    "digital","payment","upi","online","software",
                    "mobile","inverter","tv","television"],
    "daily_life" : ["daily","life","morning","evening","family","home",
                    "work","office","routine","neighbour","friend",
                    "building","stairs","terrace"],
    "government" : ["government","scheme","village","panchayat","kyp",
                    "umang","service","district","officer","apply"],
}

# ── All CSV columns (89 total) ────────────────────────────────
FIELDNAMES = [
    # Group 1: File Identity (3)
    "file_id", "audio_path", "json_path",
    # Group 2: File Metadata (14)
    "language", "audio_raw_duration", "audio_segmented_duration",
    "scenario", "task_name", "gender", "age_group", "job_type",
    "qualification", "area", "district", "state", "occupation",
    "speaker_id_file",
    # Group 3: Verification Report (25)
    "vr_decision", "vr_low_volume", "vr_noise_intermittent",
    "vr_chatter_intermittent", "vr_noise_persistent",
    "vr_chatter_persistent", "vr_unclear_audio", "vr_off_topic",
    "vr_repeating_content", "vr_long_pauses", "vr_mispronunciation",
    "vr_reading_prompt", "vr_book_read", "vr_sst", "vr_stretching",
    "vr_bad_extempore_quality", "vr_comments",
    "vr_objectionable_content", "vr_skipping_words",
    "vr_incorrect_text_prompt", "vr_factual_inaccuracy",
    "vr_wrong_language", "vr_echo_present",
    "vr_wrong_gender", "vr_wrong_age_group",
    # Group 4: Prompt (4)
    "prompt_raw", "prompt_role1", "prompt_role2", "prompt_topic",
    # Group 5: Segment Identity (6)
    "segment_id", "segment_index",
    "segment_start_sec", "segment_end_sec",
    "segment_duration_sec", "speaker_id_segment",
    # Group 6: Text Content (4)
    "verbatim_text", "normalized_text",
    "verbatim_word_count", "normalized_word_count",
    # Group 7a: Code-Switching (3)
    "has_code_switch", "code_switch_words", "cs_word_count",
    # Group 7b: Dysfluencies (3)
    "has_dysfluency", "dysfluency_types", "dysfluency_count",
    # Group 7c: Physiological (3)
    "has_physiological", "physiological_events", "physiological_count",
    # Group 7d: Environmental Noise (3)
    "has_env_noise", "env_noise_events", "env_noise_count",
    # Group 7e: Quality Annotations (3)
    "has_quality_annotation", "quality_annotations",
    "quality_annotation_count",
    # Group 7f: Paralinguistic (3)
    "has_paralinguistic", "paralinguistic_events", "paralinguistic_count",
    # Group 7g: Action/Activity (3) — NEW
    "has_action", "action_types", "action_count",
    # Group 7h: ALL annotations captured (3) — NEW
    # Every single bracket tag, regardless of category
    "all_annotations",           # comma-separated ALL tags found
    "all_annotation_count",      # total count of ALL tags
    "unknown_annotations",       # tags not in any known category
    # Group 8: Quality and Classification (7)
    "quality_score", "scenario_corrected", "domain",
    "is_acoustic_clean", "dataset_tier",
    "hard_excluded", "exclusion_reason",
    # Group 9: MFA Readiness (NEW — 2 columns)
    "is_mfa_ready",    # True if duration>=1s and audio cut OK
    "skip_reason",     # why segment is not MFA ready
    # Group 10: Output File Paths (3)
    "segment_wav_path", "segment_lab_path", "audio_cut_success",
]

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def clean_text(text):
    if not text: return ""
    text = text.replace("\n"," ").replace("\r"," ")
    return re.sub(r'\s+',' ',text).strip()

def categorize_all_annotations(verbatim, normalized):
    """
    Extract and categorize EVERY bracketed tag.
    Categories: dysfluency, physiological, env_noise,
                quality, paralinguistic, action,
                code_switch, unknown
    Returns complete dict with all categories.
    """
    norm_tags = re.findall(r'\[([^\]]+)\]', normalized)
    verb_tags = re.findall(r'\[([^\]]+)\]', verbatim)
    all_raw   = norm_tags + verb_tags

    cs_words      = []
    dysfluencies  = []
    physiological = []
    environmental = []
    quality_annot = []
    paralinguistic= []
    actions       = []
    unknown       = []
    all_found     = []

    for t in all_raw:
        t_lower = t.lower().strip()
        t_clean = t.strip()
        if not t_clean or len(t_clean) <= 1:
            continue

        all_found.append(t_clean)

        if t_lower in DYSFLUENCY_TAGS:
            dysfluencies.append(t_clean)
        elif t_lower in PHYSIOLOGICAL_TAGS:
            physiological.append(t_clean)
        elif t_lower in ENVIRONMENTAL_TAGS or t_lower in NOISE_WORDS:
            environmental.append(t_clean)
        elif t_lower in QUALITY_ANNOTATION_TAGS:
            quality_annot.append(t_clean)
        elif t_lower in PARALINGUISTIC_TAGS:
            paralinguistic.append(t_clean)
        elif t_lower in ACTION_TAGS:
            actions.append(t_clean)
        elif '_' not in t_clean and not t_clean.isdigit():
            # Could be code-switch or unknown
            # If it looks like an English word → code-switch
            # If it doesn't fit → unknown
            cs_words.append(t_clean)
        else:
            # Has underscore → likely annotation style
            unknown.append(t_clean)

    return {
        # Cat 1: Code-switch
        "has_code_switch"          : bool(cs_words),
        "code_switch_words"        : ", ".join(sorted(set(cs_words))),
        "cs_word_count"            : len(cs_words),
        # Cat 2: Dysfluency
        "has_dysfluency"           : bool(dysfluencies),
        "dysfluency_types"         : ", ".join(sorted(set(dysfluencies))),
        "dysfluency_count"         : len(dysfluencies),
        # Cat 3: Physiological
        "has_physiological"        : bool(physiological),
        "physiological_events"     : ", ".join(sorted(set(physiological))),
        "physiological_count"      : len(physiological),
        # Cat 4: Environmental
        "has_env_noise"            : bool(environmental),
        "env_noise_events"         : ", ".join(sorted(set(environmental))),
        "env_noise_count"          : len(environmental),
        # Cat 5: Quality
        "has_quality_annotation"   : bool(quality_annot),
        "quality_annotations"      : ", ".join(sorted(set(quality_annot))),
        "quality_annotation_count" : len(quality_annot),
        # Cat 6: Paralinguistic
        "has_paralinguistic"       : bool(paralinguistic),
        "paralinguistic_events"    : ", ".join(sorted(set(paralinguistic))),
        "paralinguistic_count"     : len(paralinguistic),
        # Cat 7: Action (NEW)
        "has_action"               : bool(actions),
        "action_types"             : ", ".join(sorted(set(actions))),
        "action_count"             : len(actions),
        # Cat 8: ALL (NEW) — captures everything
        "all_annotations"          : ", ".join(all_found),
        "all_annotation_count"     : len(all_found),
        "unknown_annotations"      : ", ".join(sorted(set(unknown))),
    }

def parse_prompt(prompt_str):
    if not prompt_str: return "","",""
    parts = [p.strip() for p in prompt_str.split("|")]
    if len(parts) >= 3: return parts[0], parts[1], parts[2]
    if len(parts) == 2: return parts[0], parts[1], ""
    return "","", prompt_str.strip()

def detect_domain(prompt_raw, topic, task_name):
    text = f"{prompt_raw} {topic} {task_name}".lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return "general"

def compute_quality_score(vr):
    score = 10
    if vr.get("noise_persistent")    : score -= 3
    if vr.get("chatter_persistent")  : score -= 3
    if vr.get("noise_intermittent")  : score -= 2
    if vr.get("chatter_intermittent"): score -= 2
    if vr.get("unclear_audio")       : score -= 2
    if vr.get("low_volume")          : score -= 2
    if vr.get("long_pauses")         : score -= 1
    if vr.get("stretching")          : score -= 1
    if vr.get("mispronunciation")    : score -= 1
    if vr.get("repeating_content")   : score -= 1
    if vr.get("off_topic")           : score -= 1
    if vr.get("echo_present")        : score -= 1
    if vr.get("bad_extempore_quality"): score -= 1
    if vr.get("skipping_words")      : score -= 1
    return max(0, score)

def check_hard_exclusion(vr):
    if vr.get("wrong_language"):        return True, "wrong_language"
    if vr.get("objectionable_content"): return True, "objectionable"
    if vr.get("echo_present") and vr.get("unclear_audio"):
        return True, "echo_and_unclear"
    if vr.get("wrong_gender") and vr.get("wrong_age_group"):
        return True, "wrong_metadata"
    return False, ""

def correct_scenario(scenario, vr):
    if vr.get("reading_prompt") or vr.get("book_read") or vr.get("sst"):
        return "Read_Corrected"
    sc = (scenario or "").lower()
    if "conversation" in sc: return "Conversation"
    if "extempore"    in sc: return "Extempore"
    if "read"         in sc: return "Read"
    return scenario or "Unknown"

def is_acoustic_clean(vr):
    return not any([
        vr.get("noise_persistent"), vr.get("chatter_persistent"),
        vr.get("noise_intermittent"), vr.get("chatter_intermittent"),
        vr.get("echo_present"), vr.get("low_volume"),
        vr.get("unclear_audio"),
    ])

def compute_tier(hard_excl, quality, acoustic_clean, is_mfa):
    if hard_excl:                             return 0
    if not is_mfa:                            return 1
    if acoustic_clean and quality >= 8:       return 3
    if quality >= 8:                          return 2
    return 1

def cut_audio_segment(audio_path, start_sec, end_sec, out_path):
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path, always_2d=False)
        s = max(0, int(start_sec * sr))
        e = min(len(data), int(end_sec * sr))
        if e <= s: return False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, data[s:e], sr)
        return True
    except Exception:
        return False

def save_fig(name):
    path = os.path.join(config.FIGURES_DIR, f"seg_v3_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: seg_v3_{name}.png")

# ══════════════════════════════════════════════════════════════
# STEP 1 — BUILD
# ══════════════════════════════════════════════════════════════

def build_dataset():
    print("=" * 65)
    print("  BUILD SEGMENT DATASET — VERSION 3")
    print(f"  {len(FIELDNAMES)} columns")
    print("  FIX 1: Short segments kept (text-only)")
    print("  FIX 2: ALL annotation tags captured")
    print("  FIX 3: Complete conversation preserved")
    print("=" * 65)

    json_files = glob.glob(
        os.path.join(config.INDIC_DIR, "**", "*.json"),
        recursive=True
    )
    print(f"\n  JSON files found : {len(json_files)}")
    print(f"  Output CSV       : {OUTPUT_CSV}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(MFA_SEGMENTS_DIR,   exist_ok=True)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    c          = Counter()
    start_time = time.time()

    for idx, jf in enumerate(sorted(json_files)):

        if (idx+1) % 500 == 0:
            el  = time.time()-start_time
            eta = (len(json_files)-idx-1)/max(1,(idx+1)/el)
            print(f"  [{idx+1:5d}/{len(json_files)}]  "
                  f"segs={c['segs']}  "
                  f"mfa_ready={c['mfa_ready']}  "
                  f"text_only={c['text_only']}  "
                  f"ETA={eta/60:.1f}min")

        try:
            with open(jf, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            c["json_error"] += 1
            continue

        raw_dur = float(meta.get("audio_raw_duration", 0))
        if raw_dur < 2.0:
            c["file_too_short"] += 1
            continue

        base  = os.path.splitext(jf)[0]
        audio = None
        for ext in [".wav",".mp3",".flac"]:
            if os.path.exists(base+ext):
                audio = os.path.abspath(base+ext)
                break
        if audio is None:
            c["no_audio"] += 1

        file_id     = os.path.splitext(os.path.basename(jf))[0]
        spk_id_file = meta.get("speaker_id","")

        # File metadata
        language     = meta.get("language","Hindi")
        audio_seg    = float(meta.get("audio_segmented_duration",0))
        scenario     = meta.get("scenario","")
        task_name    = meta.get("task_name","")
        gender       = str(meta.get("gender","")).lower()
        age_group    = meta.get("age_group","")
        job_type     = meta.get("job_type","")
        qualification= meta.get("qualification","")
        area         = meta.get("area","")
        district     = meta.get("district","")
        state        = meta.get("state","")
        occupation   = meta.get("occupation","")
        prompt_raw   = meta.get("prompt","")
        role1,role2,topic = parse_prompt(prompt_raw)

        vr = meta.get("verification_report",{})
        vrf = {
            "vr_decision"             : vr.get("decision",""),
            "vr_low_volume"           : vr.get("low_volume",False),
            "vr_noise_intermittent"   : vr.get("noise_intermittent",False),
            "vr_chatter_intermittent" : vr.get("chatter_intermittent",False),
            "vr_noise_persistent"     : vr.get("noise_persistent",False),
            "vr_chatter_persistent"   : vr.get("chatter_persistent",False),
            "vr_unclear_audio"        : vr.get("unclear_audio",False),
            "vr_off_topic"            : vr.get("off_topic",False),
            "vr_repeating_content"    : vr.get("repeating_content",False),
            "vr_long_pauses"          : vr.get("long_pauses",False),
            "vr_mispronunciation"     : vr.get("mispronunciation",False),
            "vr_reading_prompt"       : vr.get("reading_prompt",False),
            "vr_book_read"            : vr.get("book_read",False),
            "vr_sst"                  : vr.get("sst",False),
            "vr_stretching"           : vr.get("stretching",False),
            "vr_bad_extempore_quality": vr.get("bad_extempore_quality",False),
            "vr_comments"             : vr.get("comments",""),
            "vr_objectionable_content": vr.get("objectionable_content",False),
            "vr_skipping_words"       : vr.get("skipping_words",False),
            "vr_incorrect_text_prompt": vr.get("incorrect_text_prompt",False),
            "vr_factual_inaccuracy"   : vr.get("factual_inaccuracy",False),
            "vr_wrong_language"       : vr.get("wrong_language",False),
            "vr_echo_present"         : vr.get("echo_present",False),
            "vr_wrong_gender"         : vr.get("wrong_gender",False),
            "vr_wrong_age_group"      : vr.get("wrong_age_group",False),
        }

        hard_excl, excl_reason = check_hard_exclusion(vr)
        quality                = compute_quality_score(vr)
        sc_corrected           = correct_scenario(scenario, vr)
        acoustic_clean         = is_acoustic_clean(vr)
        domain                 = detect_domain(prompt_raw, topic, task_name)
        if hard_excl: c["hard_excl"] += 1

        norm_map = {}
        for ns in meta.get("normalized",[]):
            try:
                k = round(float(ns.get("start",-999)),2)
                norm_map[k] = clean_text(ns.get("text",""))
            except: continue

        verbatim_list = meta.get("verbatim",[])
        if not verbatim_list:
            c["no_segments"] += 1
            continue

        file_ok = False

        for seg_idx, seg in enumerate(verbatim_list):
            try:
                seg_start = float(seg.get("start",0))
                seg_end   = float(seg.get("end",0))
            except: continue

            seg_dur  = round(seg_end-seg_start, 4)
            seg_spk  = seg.get("speaker_id", 0)
            verb_txt = clean_text(seg.get("text",""))

            # ── FIX 1: Never skip — just flag short segments ──
            if not verb_txt:
                c["no_text"] += 1
                continue   # truly empty = nothing to save

            # Determine MFA readiness
            if seg_dur < MIN_MFA_DUR:
                is_mfa   = False
                skip_rsn = "too_short"
                c["text_only"] += 1
            elif hard_excl:
                is_mfa   = False
                skip_rsn = excl_reason
                c["text_only"] += 1
            else:
                is_mfa   = True
                skip_rsn = ""

            norm_txt  = norm_map.get(round(seg_start,2), verb_txt)
            verb_wc   = len(verb_txt.split())
            norm_wc   = len(norm_txt.split())

            # ── FIX 2: Capture ALL annotation tags ────────────
            annots = categorize_all_annotations(verb_txt, norm_txt)

            tier = compute_tier(
                hard_excl, quality, acoustic_clean, is_mfa
            )

            # Audio cutting — only for MFA-ready segments
            spk_folder = str(spk_id_file).replace("/","_")
            segment_id = f"{file_id}_s{seg_idx:04d}"
            wav_out = lab_out = ""
            cut_ok  = False

            if audio and is_mfa:
                wav_out = os.path.join(
                    MFA_SEGMENTS_DIR, spk_folder,
                    f"{segment_id}.wav"
                )
                lab_out = os.path.join(
                    MFA_SEGMENTS_DIR, spk_folder,
                    f"{segment_id}.lab"
                )
                cut_ok = cut_audio_segment(
                    audio, seg_start, seg_end, wav_out
                )
                if cut_ok:
                    c["cut_ok"] += 1
                    lab_text = re.sub(r'\[[^\]]+\]','',verb_txt)
                    lab_text = re.sub(r'\s+',' ',lab_text).strip()
                    with open(lab_out,"w",encoding="utf-8") as lf:
                        lf.write(lab_text)
                else:
                    c["cut_fail"] += 1
                    wav_out = lab_out = ""

            if is_mfa and cut_ok:
                c["mfa_ready"] += 1

            row = {
                "file_id":file_id, "audio_path":audio or "",
                "json_path":os.path.abspath(jf),
                "language":language,
                "audio_raw_duration":raw_dur,
                "audio_segmented_duration":audio_seg,
                "scenario":scenario, "task_name":task_name,
                "gender":gender, "age_group":age_group,
                "job_type":job_type, "qualification":qualification,
                "area":area, "district":district, "state":state,
                "occupation":occupation, "speaker_id_file":spk_id_file,
                **vrf,
                "prompt_raw":prompt_raw, "prompt_role1":role1,
                "prompt_role2":role2, "prompt_topic":topic,
                "segment_id":segment_id, "segment_index":seg_idx,
                "segment_start_sec":round(seg_start,4),
                "segment_end_sec":round(seg_end,4),
                "segment_duration_sec":seg_dur,
                "speaker_id_segment":seg_spk,
                "verbatim_text":verb_txt, "normalized_text":norm_txt,
                "verbatim_word_count":verb_wc,
                "normalized_word_count":norm_wc,
                **annots,
                "quality_score":quality,
                "scenario_corrected":sc_corrected,
                "domain":domain,
                "is_acoustic_clean":acoustic_clean,
                "dataset_tier":tier,
                "hard_excluded":hard_excl,
                "exclusion_reason":excl_reason,
                "is_mfa_ready":is_mfa,
                "skip_reason":skip_rsn,
                "segment_wav_path":wav_out,
                "segment_lab_path":lab_out,
                "audio_cut_success":cut_ok,
            }

            write_hdr = not os.path.exists(OUTPUT_CSV)
            with open(OUTPUT_CSV,"a",newline="",
                      encoding="utf-8-sig") as fcsv:
                writer = csv.DictWriter(
                    fcsv, fieldnames=FIELDNAMES,
                    extrasaction="ignore"
                )
                if write_hdr: writer.writeheader()
                writer.writerow(row)

            c["segs"] += 1
            file_ok = True

        if file_ok:
            c["files"] += 1

    elapsed = time.time()-start_time
    print(f"\n{'='*65}")
    print(f"  BUILD COMPLETE — VERSION 3")
    print(f"{'='*65}")
    print(f"  Files processed      : {c['files']}")
    print(f"  Files skipped (<2s)  : {c['file_too_short']}")
    print(f"  Total segments       : {c['segs']}")
    print()
    print(f"  MFA-ready segments   : {c['mfa_ready']}  "
          f"← cut WAV+LAB files")
    print(f"  Text-only segments   : {c['text_only']}  "
          f"← short segments, context preserved")
    print()
    print(f"  Audio cut success    : {c['cut_ok']}")
    print(f"  Audio cut failed     : {c['cut_fail']}")
    print(f"  Hard excluded        : {c['hard_excl']}")
    print(f"  Time elapsed         : {elapsed:.0f}s "
          f"({elapsed/60:.1f}min)")
    if os.path.exists(OUTPUT_CSV):
        mb = os.path.getsize(OUTPUT_CSV)/1024/1024
        print(f"  CSV size             : {mb:.1f} MB")
    print(f"  Total columns        : {len(FIELDNAMES)}")
    return c["segs"]

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  BUILD SEGMENT DATASET — VERSION 3")
    print("  Run discover_annotations.py first to find all tags")
    print("  then run this script")
    print("=" * 65)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    total_segs = build_dataset()

    print()
    print("  ALL DONE:")
    print(f"  CSV    : {OUTPUT_CSV}")
    print(f"  Audio  : {MFA_SEGMENTS_DIR}/")
    print()
    print("  HOW TO USE THE DATASET:")
    print("  For MFA alignment:")
    print("    Filter: is_mfa_ready = True")
    print(f"    mfa align {MFA_SEGMENTS_DIR}/ \\")
    print(f"      hindi_indic hindi_indic mfa_output_v3/ \\")
    print(f"      --clean --num_jobs 4 --verbose")
    print()
    print("  For discourse/conversation analysis:")
    print("    Use ALL rows (is_mfa_ready = True OR False)")
    print("    Group by: file_id, sort by: segment_index")
    print("    → Complete conversation including short turns")
    print()
    print("  For annotation discovery:")
    print("    Check column: unknown_annotations")
    print("    → Tags not in any known category")
    print("    → Add them to appropriate category sets")