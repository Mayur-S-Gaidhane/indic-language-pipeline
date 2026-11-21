#!/usr/bin/env python3
"""
run_pipeline_v2.py — Indic Language Pipeline V2

Integrated pipeline: Acoustic + Linguistic + Stanza CoNLL-U

Takes Hindi AI4Bharat IndicVoices JSON + WAV files as input.
Runs all phases automatically including MFA.

Produces three output files:
  segments.csv          (129 cols) — segment-level features
  word_features.csv     (128 cols) — word-level features
                                     (95 original + 15 acoustic
                                      + 18 linguistic)
  stanza_output.conllu              — CoNLL-U dependency parse

Usage:
  python run_pipeline_v2.py --input_file /path/to/file.json --run_name demo
  python run_pipeline_v2.py --input_files f1.json f2.json --run_name test
  python run_pipeline_v2.py --input_dir input_data/train/ --run_name full

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import sys
import csv
import json
import time
import math
import argparse
import subprocess
import re
import warnings
from pathlib import Path
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# ── Check required libraries ──────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import soundfile as sf
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("WARNING: soundfile not installed — audio cutting disabled")

try:
    import parselmouth
    from parselmouth.praat import call
    PARSEL_OK = True
except ImportError:
    PARSEL_OK = False
    print("WARNING: parselmouth not installed — acoustic features disabled")

try:
    import stanza
    STANZA_OK = True
except ImportError:
    STANZA_OK = False
    print("WARNING: stanza not installed — linguistic features disabled")

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    GPT2_OK = True
except ImportError:
    GPT2_OK = False
    print("WARNING: transformers not installed — surprisal disabled")

# ── MFA settings ──────────────────────────────────────────────
MFA_ACOUSTIC   = "hindi_indic"
MFA_DICTIONARY = "hindi_indic"
MFA_JOBS       = 4
MIN_DUR        = 1.0
MAX_DUR        = 60.0
GPT2_MODEL     = "surajp/gpt2-hindi"

# ── Annotation taxonomy ───────────────────────────────────────
DYSFLUENCY_TAGS = {
    "stammers","stammer","stutter","stutters",
    "uhh","uh","umm","um","hmm","hm",
    "err","erm","ahh","ah","ahm",
    "stammering","stuttering","filled_pause","false_start",
    "ugh","uhm","ehh","eh","aah",
}
PHYSIOLOGICAL_TAGS = {
    "inhaling","exhaling","breathing","breath",
    "cough","coughing","sneeze","sneezing",
    "yawning","yawn","swallowing","swallow",
    "sigh","sighing","throat_clearing","clearing_throat",
    "lip_smacking","clicking","nose_blowing",
    "sniffling","sniff","hiccup","hiccups",
}
ENV_NOISE_TAGS = {
    "noise","background_noise","env_noise",
    "traffic","traffic_noise","horn","honking",
    "phone_ringing","phone_vibrating","bell","doorbell",
    "knock","knocking","tv_sound","tv_noise",
    "barking","dog_barking","bird","crow",
    "crowd","crowd_noise","clinking","buzzer",
    "construction","drilling","vehicle_noise","ac_noise",
}
QUALITY_TAGS = {
    "unintelligible","inaudible","unclear_speech",
    "noise_here","crosstalk","overlap","unknown_word",
}
PARALINGUISTIC_TAGS = {
    "laughing","laughter","laugh","giggling","giggle",
    "chuckling","chuckle","crying","cry","sobbing","sob",
    "whispering","whisper","shouting","yelling",
    "singing","humming","hum","groaning","groan",
}
ACTION_TAGS = {
    "clapping","clap","tapping","tap","snapping","snap",
    "talking","speaking","mumbling","mumble","pausing",
}

# ── Domain keywords ───────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "travel"    : ["location","travel","trip","tour","destination",
                   "hotel","flight","train","bus","taxi","route",
                   "rickshaw","driver","passenger","road","map",
                   "visit","place","city","mandir","fort","tourist"],
    "medical"   : ["doctor","patient","hospital","medicine","disease",
                   "symptom","treatment","health","clinic","nurse",
                   "prescription","fever","pain","surgery","medical"],
    "commerce"  : ["product","buy","sell","price","shop","market",
                   "order","delivery","payment","discount","item",
                   "customer","review","refund","bigbasket","farmer",
                   "vegetable","kilo","rupee","supermarket","bank"],
    "education" : ["teacher","student","school","college","subject",
                   "exam","study","class","concept","explain",
                   "university","degree","course","lesson"],
    "food"      : ["food","restaurant","dish","recipe","eat","cook",
                   "menu","taste","swiggy","zomato","meal","icecream"],
    "technology": ["phone","app","internet","computer","alexa","siri",
                   "digital","payment","upi","online","software",
                   "mobile","inverter","tv","television"],
    "daily_life": ["daily","life","morning","evening","family","home",
                   "work","office","routine","neighbour","friend",
                   "building","stairs","terrace"],
    "government": ["government","scheme","village","panchayat","kyp",
                   "umang","service","district","officer","apply"],
}

# ── Regex patterns ────────────────────────────────────────────
BRACKET_RE     = re.compile(r'\[\s*([^\]]*?)\s*\]')
CLEAN_RE       = re.compile(r'\[\s*[^\]]*?\s*\]')
HYPHEN_RE      = re.compile(r'(?<=\S)-(?=\S)')
PUNCT_RE       = re.compile(
    r'[,?!.;:"\'\u2018\u2019\u201c\u201d\u0964\u0965$`|/]')
ZW_RE          = re.compile(r'[\u200b\u200c\u200d\ufeff]')
BRACKET_TAG_RE = re.compile(r'\[\s*[^\]]*?\s*\]')
UNMATCHED_O_RE = re.compile(r'[\[(]\s*\S*')
UNMATCHED_C_RE = re.compile(r'\S*\s*[\])]')

def log(msg):
    print(f"  {msg}", flush=True)

def banner(msg):
    print(f"\n{'='*60}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{'='*60}", flush=True)

# ── Annotation helpers ────────────────────────────────────────
def extract_annotations(text):
    tags = BRACKET_RE.findall(text)
    result = {
        "dysfluency":[],"physiological":[],"env_noise":[],
        "quality":[],"paralinguistic":[],"action":[],
        "code_switch":[],"unknown":[]
    }
    for tag in tags:
        t = tag.strip().lower()
        if t in DYSFLUENCY_TAGS:       result["dysfluency"].append(t)
        elif t in PHYSIOLOGICAL_TAGS:  result["physiological"].append(t)
        elif t in ENV_NOISE_TAGS:      result["env_noise"].append(t)
        elif t in QUALITY_TAGS:        result["quality"].append(t)
        elif t in PARALINGUISTIC_TAGS: result["paralinguistic"].append(t)
        elif t in ACTION_TAGS:         result["action"].append(t)
        elif re.match(r'^[a-zA-Z]', t):result["code_switch"].append(t)
        else:                          result["unknown"].append(t)
    return result

def clean_hindi(text):
    t = CLEAN_RE.sub('', text)
    return re.sub(r'\s+', ' ', t).strip()

def count_words(text):
    t = CLEAN_RE.sub('', text)
    t = re.sub(r'\s+', ' ', t).strip()
    return len(t.split()) if t else 0

def compute_quality_score(vr):
    score = 10
    if vr.get("noise_persistent"):     score -= 3
    if vr.get("chatter_persistent"):   score -= 3
    if vr.get("noise_intermittent"):   score -= 2
    if vr.get("chatter_intermittent"): score -= 2
    if vr.get("unclear_audio"):        score -= 2
    if vr.get("low_volume"):           score -= 2
    if vr.get("long_pauses"):          score -= 1
    if vr.get("stretching"):           score -= 1
    if vr.get("mispronunciation"):     score -= 1
    if vr.get("repeating_content"):    score -= 1
    if vr.get("off_topic"):            score -= 1
    if vr.get("echo_present"):         score -= 1
    if vr.get("bad_extempore_quality"):score -= 1
    if vr.get("skipping_words"):       score -= 1
    return max(0, score)

def check_hard_exclusion(vr):
    if vr.get("wrong_language"):        return True, "wrong_language"
    if vr.get("objectionable_content"): return True, "objectionable"
    if vr.get("echo_present") and vr.get("unclear_audio"):
        return True, "echo_and_unclear"
    if vr.get("wrong_gender") and vr.get("wrong_age_group"):
        return True, "wrong_metadata"
    return False, ""

def is_acoustic_clean(vr):
    return not any([
        vr.get("noise_persistent"), vr.get("chatter_persistent"),
        vr.get("noise_intermittent"), vr.get("chatter_intermittent"),
        vr.get("echo_present"), vr.get("low_volume"),
        vr.get("unclear_audio"),
    ])

def compute_tier(hard_excl, quality, acoustic_clean, is_mfa):
    if hard_excl:                       return 0
    if not is_mfa:                      return 1
    if acoustic_clean and quality >= 8: return 3
    if quality >= 8:                    return 2
    return 1

def correct_scenario(scenario, vr):
    if vr.get("reading_prompt") or vr.get("book_read") or vr.get("sst"):
        return "Read_Corrected"
    sc = (scenario or "").lower()
    if "conversation" in sc: return "Conversation"
    if "extempore"    in sc: return "Extempore"
    if "read"         in sc: return "Read"
    return scenario

def detect_domain(prompt_raw, topic, task_name):
    text = (prompt_raw + " " + topic + " " + task_name).lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return "general"

def compute_quality_v4(quality_score, dysfluency_density):
    acoustic = int(quality_score or 10)
    if dysfluency_density > 0.30:   ling_ded = 2
    elif dysfluency_density > 0.15: ling_ded = 1
    else:                           ling_ded = 0
    return {
        "quality_score_v4"            : max(0, acoustic - ling_ded),
        "quality_acoustic_component"  : acoustic,
        "quality_linguistic_deduction": ling_ded,
        "dysfluency_density"          : dysfluency_density,
    }

def parse_prompt(prompt_str):
    parts = [p.strip() for p in prompt_str.split("|")]
    if len(parts) >= 3: return parts[0], parts[1], parts[2]
    if len(parts) == 2: return parts[0], parts[1], ""
    return "", "", prompt_str.strip()

def clean_lab_text(text):
    t = BRACKET_TAG_RE.sub('', text)
    t = HYPHEN_RE.sub(' ', t)
    t = PUNCT_RE.sub('', t)
    t = ZW_RE.sub('', t)
    tokens = []
    for tok in t.split():
        if '[' in tok or '(' in tok: tok = UNMATCHED_O_RE.sub('', tok)
        if ']' in tok or ')' in tok: tok = UNMATCHED_C_RE.sub('', tok)
        tok = tok.strip()
        if tok: tokens.append(tok)
    return ' '.join(tokens)

# ── IC and SC computation (matches Prof's dlt.py) ────────────
def compute_ic_sc(words):
    """Compute Integration Cost and Storage Cost per word.

    Follows Prof. Sidharth Ranjan's DLT implementation (dlt.py):
      IC: baseline 1 per token + count of intervening non-PUNCT words
          between dependent and head, added to the later token.
      SC: baseline 1 + count of open (unresolved) dependencies at
          each position, using condition d <= i < h.
    Punctuation tokens get IC=0 and SC=0.
    """
    n = len(words)
    if n == 0:
        return [], []

    # ── Integration Cost ──────────────────────────────────────
    # Every token starts with baseline IC = 1
    ic_list = [1] * n

    for j, w in enumerate(words):
        dep_id = j          # 0-indexed position of dependent
        head_id = w.head - 1  # convert 1-indexed head to 0-indexed
        if w.head == 0:     # root — no dependency to count
            continue
        if head_id < 0 or head_id >= n:  # dangling head
            continue

        lower = min(dep_id, head_id)
        upper = max(dep_id, head_id)
        later = upper  # the token that occurs later in sentence

        # Count intervening tokens that are NOT punctuation
        intervening = sum(
            1 for k in range(lower + 1, upper)
            if words[k].upos != "PUNCT"
        )

        # Add intervening count to the later token
        ic_list[later] += intervening

    # Force punctuation IC to 0
    for i, w in enumerate(words):
        if w.upos == "PUNCT":
            ic_list[i] = 0

    # ── Storage Cost ──────────────────────────────────────────
    # Collect valid (dep, head) pairs as 0-indexed positions
    valid_ids = set(range(n))
    dependencies = []
    for j, w in enumerate(words):
        head_id = w.head - 1
        if w.head != 0 and head_id in valid_ids:
            dependencies.append((j, head_id))

    sc_list = [0] * n
    for i in range(n):
        # Count dependencies where d <= i < h (using 0-indexed)
        unresolved = sum(
            1 for (d, h) in dependencies
            if d <= i < h
        )
        sc_list[i] = 1 + unresolved

    # Force punctuation SC to 0
    for i, w in enumerate(words):
        if w.upos == "PUNCT":
            sc_list[i] = 0

    return ic_list, sc_list

# ── PHASE 1: Parse JSON and cut audio ────────────────────────
def phase1_parse_and_cut(json_path, mfa_seg_dir):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    file_id   = Path(json_path).stem
    wav_path  = str(json_path).replace('.json', '.wav')
    if not os.path.exists(wav_path):
        log(f"ERROR: WAV not found: {wav_path}")
        return []

    language  = data.get("language","Hindi")
    scenario  = data.get("scenario","")
    task_name = data.get("task_name","")
    gender    = data.get("gender","")
    age_group = data.get("age_group","")
    job_type  = data.get("job_type","")
    qual      = data.get("qualification","")
    area      = data.get("area","")
    district  = data.get("district","")
    state     = data.get("state","")
    occupation= data.get("occupation","")
    speaker_id= data.get("speaker_id","")
    prompt_raw= data.get("prompt","")
    audio_raw_dur = data.get("audio_raw_duration",0)
    audio_seg_dur = data.get("audio_segmented_duration",0)
    vr        = data.get("verification_report",{})
    verbatim  = data.get("verbatim",[])
    normalized= data.get("normalized",[])

    prompt_role1, prompt_role2, prompt_topic = parse_prompt(prompt_raw)
    hard_excl, excl_reason = check_hard_exclusion(vr)
    quality        = compute_quality_score(vr)
    acoustic_clean = is_acoustic_clean(vr)
    domain         = detect_domain(prompt_raw, prompt_topic, task_name)
    scen_corrected = correct_scenario(scenario, vr)

    if AUDIO_OK:
        try:
            audio_data, sr = sf.read(wav_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
        except:
            audio_data, sr = None, 16000
    else:
        audio_data, sr = None, 16000

    spk_dir = os.path.join(mfa_seg_dir, speaker_id)
    os.makedirs(spk_dir, exist_ok=True)
    segments = []

    for i, vb in enumerate(verbatim):
        seg_id    = f"{file_id}_s{i:04d}"
        start_sec = float(vb.get("start",0))
        end_sec   = float(vb.get("end",0))
        duration  = round(end_sec - start_sec, 3)
        vb_text   = vb.get("text","").strip()
        nm_text   = normalized[i].get("text","").strip() \
                    if i < len(normalized) else vb_text

        vb_ann  = extract_annotations(vb_text)
        nm_ann  = extract_annotations(nm_text)
        clean_text = clean_hindi(nm_text)
        vb_wc   = count_words(vb_text)
        nm_wc   = count_words(nm_text)
        clean_wc= count_words(clean_text)
        dysfl_count = len(vb_ann["dysfluency"])
        dysfl_den   = round(dysfl_count/vb_wc, 4) if vb_wc > 0 else 0.0
        qs_v4   = compute_quality_v4(quality, dysfl_den)

        wav_out = os.path.join(spk_dir, f"{seg_id}.wav")
        lab_out = os.path.join(spk_dir, f"{seg_id}.lab")
        is_mfa_ready = False
        skip_reason  = ""
        cut_success  = False

        if hard_excl:
            skip_reason = excl_reason
            wav_out = lab_out = ""
        elif duration < MIN_DUR:
            skip_reason = "too_short"
            wav_out = lab_out = ""
        elif duration > MAX_DUR:
            skip_reason = "too_long"
            wav_out = lab_out = ""
        else:
            lab_text = clean_lab_text(vb_text)
            if not lab_text.strip():
                skip_reason = "empty_after_cleaning"
                wav_out = lab_out = ""
            else:
                if AUDIO_OK and audio_data is not None:
                    try:
                        s = max(0, int(start_sec * sr))
                        e = min(len(audio_data), int(end_sec * sr))
                        if e > s:
                            seg_audio = audio_data[s:e]
                            sf.write(wav_out, seg_audio, sr)
                            with open(lab_out,'w',encoding='utf-8') as lf:
                                lf.write(lab_text)
                            is_mfa_ready = True
                            cut_success  = True
                        else:
                            skip_reason = "zero_length_audio"
                            wav_out = lab_out = ""
                    except Exception as ex:
                        skip_reason = "audio_cut_error"
                        wav_out = lab_out = ""

        tier = compute_tier(hard_excl, quality, acoustic_clean, is_mfa_ready)

        seg = {
            "file_id": file_id, "audio_path": wav_path,
            "json_path": str(json_path), "language": language,
            "audio_raw_duration": audio_raw_dur,
            "audio_segmented_duration": audio_seg_dur,
            "scenario": scenario, "task_name": task_name,
            "gender": gender, "age_group": age_group,
            "job_type": job_type, "qualification": qual,
            "area": area, "district": district,
            "state": state, "occupation": occupation,
            "speaker_id_file": speaker_id,
            "vr_decision": vr.get("decision",""),
            "vr_low_volume": vr.get("low_volume",False),
            "vr_noise_intermittent": vr.get("noise_intermittent",False),
            "vr_chatter_intermittent": vr.get("chatter_intermittent",False),
            "vr_noise_persistent": vr.get("noise_persistent",False),
            "vr_chatter_persistent": vr.get("chatter_persistent",False),
            "vr_unclear_audio": vr.get("unclear_audio",False),
            "vr_off_topic": vr.get("off_topic",False),
            "vr_repeating_content": vr.get("repeating_content",False),
            "vr_long_pauses": vr.get("long_pauses",False),
            "vr_mispronunciation": vr.get("mispronunciation",False),
            "vr_reading_prompt": vr.get("reading_prompt",False),
            "vr_book_read": vr.get("book_read",False),
            "vr_sst": vr.get("sst",False),
            "vr_stretching": vr.get("stretching",False),
            "vr_bad_extempore_quality": vr.get("bad_extempore_quality",False),
            "vr_comments": vr.get("comments",""),
            "vr_objectionable_content": vr.get("objectionable_content",False),
            "vr_skipping_words": vr.get("skipping_words",False),
            "vr_incorrect_text_prompt": vr.get("incorrect_text_prompt",False),
            "vr_factual_inaccuracy": vr.get("factual_inaccuracy",False),
            "vr_wrong_language": vr.get("wrong_language",False),
            "vr_echo_present": vr.get("echo_present",False),
            "vr_wrong_gender": vr.get("wrong_gender",False),
            "vr_wrong_age_group": vr.get("wrong_age_group",False),
            "prompt_raw": prompt_raw, "prompt_role1": prompt_role1,
            "prompt_role2": prompt_role2, "prompt_topic": prompt_topic,
            "segment_id": seg_id, "segment_index": i,
            "segment_start_sec": start_sec, "segment_end_sec": end_sec,
            "segment_duration_sec": duration,
            "speaker_id_segment": vb.get("speaker_id",0),
            "speaker_id_note": "source_data_all_zero",
            "verbatim_text": vb_text, "normalized_text": nm_text,
            "clean_hindi_text": clean_text,
            "verbatim_word_count": vb_wc,
            "normalized_word_count": nm_wc,
            "clean_hindi_word_count": clean_wc,
            "has_code_switch": len(nm_ann["code_switch"]) > 0,
            "code_switch_words": "|".join(nm_ann["code_switch"]),
            "cs_word_count": len(nm_ann["code_switch"]),
            "has_dysfluency": len(vb_ann["dysfluency"]) > 0,
            "dysfluency_types": "|".join(sorted(set(vb_ann["dysfluency"]+nm_ann["dysfluency"]))),
            "verbatim_dysfluency_types": "|".join(vb_ann["dysfluency"]),
            "normalized_dysfluency_types": "|".join(nm_ann["dysfluency"]),
            "dysfluency_count": dysfl_count,
            "verbatim_dysfluency_count": len(vb_ann["dysfluency"]),
            "normalized_dysfluency_count": len(nm_ann["dysfluency"]),
            "has_physiological": len(vb_ann["physiological"]) > 0,
            "physiological_events": "|".join(sorted(set(vb_ann["physiological"]+nm_ann["physiological"]))),
            "verbatim_physiological_events": "|".join(vb_ann["physiological"]),
            "normalized_physiological_events": "|".join(nm_ann["physiological"]),
            "physiological_count": len(vb_ann["physiological"]),
            "verbatim_physiological_count": len(vb_ann["physiological"]),
            "normalized_physiological_count": len(nm_ann["physiological"]),
            "has_env_noise": len(vb_ann["env_noise"]) > 0,
            "env_noise_events": "|".join(sorted(set(vb_ann["env_noise"]+nm_ann["env_noise"]))),
            "verbatim_env_noise_events": "|".join(vb_ann["env_noise"]),
            "normalized_env_noise_events": "|".join(nm_ann["env_noise"]),
            "env_noise_count": len(vb_ann["env_noise"]),
            "verbatim_env_noise_count": len(vb_ann["env_noise"]),
            "normalized_env_noise_count": len(nm_ann["env_noise"]),
            "has_quality_annotation": len(vb_ann["quality"]) > 0,
            "quality_annotations": "|".join(sorted(set(vb_ann["quality"]+nm_ann["quality"]))),
            "verbatim_quality_annotations": "|".join(vb_ann["quality"]),
            "normalized_quality_annotations": "|".join(nm_ann["quality"]),
            "quality_annotation_count": len(vb_ann["quality"]),
            "verbatim_quality_annotation_count": len(vb_ann["quality"]),
            "normalized_quality_annotation_count": len(nm_ann["quality"]),
            "has_paralinguistic": len(vb_ann["paralinguistic"]) > 0,
            "paralinguistic_events": "|".join(sorted(set(vb_ann["paralinguistic"]+nm_ann["paralinguistic"]))),
            "verbatim_paralinguistic_events": "|".join(vb_ann["paralinguistic"]),
            "normalized_paralinguistic_events": "|".join(nm_ann["paralinguistic"]),
            "paralinguistic_count": len(vb_ann["paralinguistic"]),
            "verbatim_paralinguistic_count": len(vb_ann["paralinguistic"]),
            "normalized_paralinguistic_count": len(nm_ann["paralinguistic"]),
            "has_action": len(vb_ann["action"]) > 0,
            "action_types": "|".join(sorted(set(vb_ann["action"]+nm_ann["action"]))),
            "verbatim_action_types": "|".join(vb_ann["action"]),
            "normalized_action_types": "|".join(nm_ann["action"]),
            "action_count": len(vb_ann["action"]),
            "verbatim_action_count": len(vb_ann["action"]),
            "normalized_action_count": len(nm_ann["action"]),
            "all_annotations": "|".join(
                [t.strip().lower() for t in BRACKET_RE.findall(vb_text)] +
                [t.strip().lower() for t in BRACKET_RE.findall(nm_text)]),
            "all_annotation_count": (
                len(BRACKET_RE.findall(vb_text)) +
                len(BRACKET_RE.findall(nm_text))),
            "verbatim_all_annotation_count": len(BRACKET_RE.findall(vb_text)),
            "normalized_all_annotation_count": len(BRACKET_RE.findall(nm_text)),
            "unknown_annotations": "|".join(sorted(set(vb_ann["unknown"]+nm_ann["unknown"]))),
            "verbatim_unknown_annotations": "|".join(vb_ann["unknown"]),
            "normalized_unknown_annotations": "|".join(nm_ann["unknown"]),
            "verbatim_unknown_annotation_count": len(vb_ann["unknown"]),
            "normalized_unknown_annotation_count": len(nm_ann["unknown"]),
            "quality_score": quality,
            "quality_score_v4": qs_v4["quality_score_v4"],
            "quality_acoustic_component": qs_v4["quality_acoustic_component"],
            "quality_linguistic_deduction": qs_v4["quality_linguistic_deduction"],
            "dysfluency_density": dysfl_den,
            "scenario_corrected": scen_corrected,
            "domain": domain,
            "is_acoustic_clean": acoustic_clean,
            "dataset_tier": tier,
            "hard_excluded": hard_excl,
            "exclusion_reason": excl_reason,
            "is_mfa_ready": is_mfa_ready,
            "skip_reason": skip_reason,
            "segment_wav_path": wav_out,
            "segment_lab_path": lab_out,
            "audio_cut_success": cut_success,
        }
        segments.append(seg)
    return segments


# ── PHASE 2: MFA validate + align ────────────────────────────
def phase2_mfa(mfa_seg_dir, mfa_aln_dir, mfa_val_dir, run_output_dir):
    banner("PHASE 2: MFA Validate + Align (automatic)")
    wav_files = list(Path(mfa_seg_dir).rglob("*.wav"))
    if not wav_files:
        log("ERROR: No WAV files found for MFA.")
        return False
    log(f"WAV files for MFA : {len(wav_files)}")
    os.makedirs(mfa_val_dir, exist_ok=True)
    os.makedirs(mfa_aln_dir, exist_ok=True)
    log("Running MFA validate ...")
    val_cmd = ["mfa","validate",mfa_seg_dir,
               MFA_ACOUSTIC,MFA_DICTIONARY,
               "--num_jobs",str(MFA_JOBS),
               "--output_directory",mfa_val_dir]
    subprocess.run(val_cmd, capture_output=True, text=True)
    log("MFA validate complete ✅")
    log("Running MFA align ...")
    aln_cmd = ["mfa","align",mfa_seg_dir,
               MFA_ACOUSTIC,MFA_DICTIONARY,mfa_aln_dir,
               "--clean","--num_jobs",str(MFA_JOBS),"--verbose"]
    aln_log = os.path.join(run_output_dir,"mfa_align.log")
    with open(aln_log,"w") as logf:
        subprocess.run(aln_cmd, stdout=logf, stderr=subprocess.STDOUT)
    tg_files = list(Path(mfa_aln_dir).rglob("*.TextGrid"))
    log(f"TextGrids produced : {len(tg_files)}")
    if not tg_files:
        log(f"WARNING: No TextGrids. Check {aln_log}")
        return False
    log("MFA align complete ✅")
    return True


# ── PHASE 3: Extract words + resolve unk ─────────────────────
def phase3_extract_words(segments, mfa_aln_dir):
    banner("PHASE 3: Extracting Word Durations from TextGrids")
    seg_map  = {s["segment_id"]:s for s in segments}
    tg_files = list(Path(mfa_aln_dir).rglob("*.TextGrid"))
    log(f"TextGrid files : {len(tg_files)}")
    TOLERANCE = 0.001
    word_rows = []
    for tg_path in tg_files:
        seg_id   = tg_path.stem
        seg_meta = seg_map.get(seg_id,{})
        try:
            words_tier = []
            phones_tier = []
            current_tier = None
            cur_xmin = cur_xmax = cur_text = None
            with open(tg_path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                ls = line.strip()
                if 'name = "words"' in ls:
                    current_tier = "words"
                    cur_xmin = cur_xmax = cur_text = None
                elif 'name = "phones"' in ls:
                    current_tier = "phones"
                    cur_xmin = cur_xmax = cur_text = None
                elif ls.startswith("xmin =") and current_tier:
                    try: cur_xmin = float(ls.split("=")[1].strip())
                    except: cur_xmin = None
                elif ls.startswith("xmax =") and current_tier:
                    try: cur_xmax = float(ls.split("=")[1].strip())
                    except: cur_xmax = None
                elif ls.startswith("text =") and current_tier:
                    cur_text = ls.split("=",1)[1].strip().strip('"')
                    if cur_xmin is not None and cur_xmax is not None and cur_xmin < cur_xmax:
                        if current_tier == "words":
                            words_tier.append((cur_xmin,cur_xmax,cur_text))
                        else:
                            phones_tier.append((cur_xmin,cur_xmax,cur_text))
                    cur_xmin = cur_xmax = cur_text = None
            word_idx = 0
            for xmin, xmax, text in words_tier:
                if not text or text in ("","sp","sil","SIL"):
                    continue
                dur_ms = round((xmax-xmin)*1000, 1)
                pcount = sum(1 for px,py,pt in phones_tier
                             if pt and pt not in ("","sp","sil","SIL")
                             and py > xmin-TOLERANCE
                             and px < xmax+TOLERANCE)
                row = {
                    "segment_id": seg_id,
                    "word_index": word_idx,
                    "word": text,
                    "word_from_lab": "",
                    "word_resolved": "",
                    "word_is_resolved": 0,
                    "word_start_sec": round(xmin,4),
                    "word_end_sec": round(xmax,4),
                    "word_duration_ms": dur_ms,
                    "phone_count": pcount,
                }
                for col in seg_meta:
                    if col not in row:
                        row[col] = seg_meta[col]
                row["unknown_annotation_count"] = ""
                word_rows.append(row)
                word_idx += 1
        except Exception as e:
            log(f"  Parse error for {seg_id}: {e}")
            continue
    log(f"Total words extracted : {len(word_rows)}")
    return word_rows


def phase3_resolve_unk(word_rows, mfa_seg_dir):
    banner("PHASE 3b: Resolving <unk> + Correcting MFA Labels")
    unk_count = sum(1 for w in word_rows if w["word"]=="<unk>")
    log(f"<unk> tokens found : {unk_count}")
    seg_words = defaultdict(list)
    for idx, w in enumerate(word_rows):
        seg_words[w["segment_id"]].append((idx,w))
    resolved = 0
    for seg_id, word_list in seg_words.items():
        lab_path = None
        for root, dirs, files in os.walk(mfa_seg_dir):
            for f in files:
                if f == f"{seg_id}.lab":
                    lab_path = os.path.join(root,f)
                    break
        lab_words = []
        if lab_path and os.path.exists(lab_path):
            with open(lab_path, encoding='utf-8') as lf:
                lab_words = lf.read().strip().split()
        for pos, (idx, w) in enumerate(word_list):
            word_from_lab = lab_words[pos] if lab_words and pos < len(lab_words) else ""
            if w["word"] == "<unk>":
                w["word_from_lab"]    = word_from_lab or "<unk>"
                w["word_resolved"]    = word_from_lab or "<unk>"
                w["word_is_resolved"] = 1 if word_from_lab else 0
                if word_from_lab: resolved += 1
            elif word_from_lab and word_from_lab != w["word"]:
                w["word_from_lab"]    = word_from_lab
                w["word_resolved"]    = word_from_lab
                w["word_is_resolved"] = 0
            else:
                w["word_from_lab"]    = word_from_lab
                w["word_resolved"]    = w["word"]
                w["word_is_resolved"] = 0
    log(f"<unk> resolved : {resolved}/{unk_count} ✅")
    return word_rows


# ── PHASE 4: Acoustic features ────────────────────────────────
def extract_word_acoustics(wav_path, word_start, word_end):
    empty = {
        "f0_mean_hz":"","f0_min_hz":"","f0_max_hz":"",
        "f0_range_hz":"","f0_std_hz":"","voiced_fraction":"",
        "rms_energy":"","rms_energy_db":"",
        "jitter":"","shimmer":"","hnr":"",
        "acoustic_status":"no_parselmouth",
    }
    if not PARSEL_OK or not AUDIO_OK:
        return empty
    try:
        audio_data, sr = sf.read(wav_path)
        if audio_data.ndim > 1: audio_data = audio_data.mean(axis=1)
        s = int(word_start * sr)
        e = int(word_end * sr)
        if e <= s or (e-s) < int(0.02*sr):
            empty["acoustic_status"] = "too_short"
            return empty
        word_audio = audio_data[s:e]
        duration   = len(word_audio) / sr
        sound      = parselmouth.Sound(word_audio, sampling_frequency=sr)
        result     = dict(empty)

        try:
            rms = sound.get_rms()
            if rms and rms > 0:
                result["rms_energy"]    = round(float(rms),6)
                result["rms_energy_db"] = round(float(10*np.log10(rms**2+1e-10)),3)
        except: pass

        try:
            pitch = sound.to_pitch(time_step=0.01,pitch_floor=75,pitch_ceiling=500)
            f0_vals = [pitch.get_value_at_time(t)
                       for t in np.arange(0,duration,0.01)
                       if pitch.get_value_at_time(t)
                       and not np.isnan(pitch.get_value_at_time(t))]
            voiced = [v for v in f0_vals if v > 0]
            if voiced:
                result["f0_mean_hz"]      = round(float(np.mean(voiced)),2)
                result["f0_min_hz"]       = round(float(np.min(voiced)),2)
                result["f0_max_hz"]       = round(float(np.max(voiced)),2)
                result["f0_range_hz"]     = round(float(np.max(voiced)-np.min(voiced)),2)
                result["f0_std_hz"]       = round(float(np.std(voiced)),2)
                result["voiced_fraction"] = round(len(voiced)/max(1,len(f0_vals)),3)
            else:
                for k in ["f0_mean_hz","f0_min_hz","f0_max_hz",
                          "f0_range_hz","f0_std_hz","voiced_fraction"]:
                    result[k] = 0
        except: pass

        if duration >= 0.05:
            try:
                pp = call(sound,"To PointProcess (periodic, cc)",75,500)
                j  = call(pp,"Get jitter (local)",0,0,0.0001,0.02,1.3)
                if j and not np.isnan(j): result["jitter"] = round(float(j),6)
            except: pass
            try:
                pp = call(sound,"To PointProcess (periodic, cc)",75,500)
                sh = call([sound,pp],"Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)
                if sh and not np.isnan(sh): result["shimmer"] = round(float(sh),6)
            except: pass
            try:
                harm = call(sound,"To Harmonicity (cc)",0.01,75,0.1,1.0)
                hnr  = call(harm,"Get mean",0,0)
                if hnr and not np.isnan(hnr): result["hnr"] = round(float(hnr),3)
            except: pass

        result["acoustic_status"] = "ok_short" if duration < 0.05 else "ok"
        return result
    except:
        empty["acoustic_status"] = "error"
        return empty


def read_pauses_from_textgrid(tg_path):
    words_tier = []
    current_tier = None
    cur_xmin = cur_xmax = cur_text = None
    with open(tg_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        ls = line.strip()
        if 'name = "words"' in ls:
            current_tier = "words"
            cur_xmin = cur_xmax = cur_text = None
        elif 'name = "phones"' in ls:
            current_tier = None
        elif current_tier == "words":
            if ls.startswith("xmin ="):
                try: cur_xmin = float(ls.split("=")[1].strip())
                except: cur_xmin = None
            elif ls.startswith("xmax ="):
                try: cur_xmax = float(ls.split("=")[1].strip())
                except: cur_xmax = None
            elif ls.startswith("text ="):
                cur_text = ls.split("=",1)[1].strip().strip('"')
                if cur_xmin is not None and cur_xmax is not None:
                    words_tier.append((cur_xmin,cur_xmax,cur_text))
                cur_xmin = cur_xmax = cur_text = None
    word_entries = [(x,y,t) for x,y,t in words_tier
                   if t and t not in ("","sp","sil","SIL")]
    pause_map = {}
    for wi,(w_xmin,w_xmax,_) in enumerate(word_entries):
        pb = 0.0
        for xmin,xmax,text in words_tier:
            if xmax <= w_xmin and (not text or text in ("","sp","sil")):
                if w_xmin - xmax < 0.001:
                    pb = (xmax-xmin)*1000
        pa = 0.0
        for xmin,xmax,text in words_tier:
            if xmin >= w_xmax and (not text or text in ("","sp","sil")):
                if xmin - w_xmax < 0.001:
                    pa = (xmax-xmin)*1000
                    break
        pause_map[wi] = (round(pb,1), round(pa,1))
    return pause_map


def phase4_acoustic(word_rows, mfa_seg_dir, mfa_aln_dir):
    banner("PHASE 4: Extracting Acoustic Features (Parselmouth)")
    if not PARSEL_OK:
        log("WARNING: Parselmouth not available — skipping acoustic features")
        for w in word_rows:
            for col in ["f0_mean_hz","f0_min_hz","f0_max_hz",
                        "f0_range_hz","f0_std_hz","voiced_fraction",
                        "rms_energy","rms_energy_db","jitter","shimmer",
                        "hnr","pause_before_ms","pause_after_ms",
                        "speech_rate","acoustic_status"]:
                w[col] = ""
        return word_rows

    # Index WAV files
    wav_index = {}
    for wp in Path(mfa_seg_dir).rglob("*.wav"):
        wav_index[wp.stem] = str(wp)
    log(f"WAV files indexed: {len(wav_index)}")

    # Index TextGrid pauses
    tg_files = list(Path(mfa_aln_dir).rglob("*.TextGrid"))
    pause_data = {}
    for tg in tg_files:
        try: pause_data[tg.stem] = read_pauses_from_textgrid(str(tg))
        except: pass
    log(f"TextGrid pause data: {len(pause_data)}")

    # Speech rate per segment
    seg_word_counts = Counter(w["segment_id"] for w in word_rows)
    seg_durations   = {}
    for w in word_rows:
        sid = w["segment_id"]
        if sid not in seg_durations:
            try: seg_durations[sid] = float(w.get("segment_duration_sec",0) or 0)
            except: seg_durations[sid] = 0
    speech_rates = {
        sid: round(seg_word_counts[sid]/d, 3) if d > 0 else 0
        for sid, d in seg_durations.items()
    }

    ok=0; short=0; err=0
    for i, w in enumerate(word_rows):
        sid        = w["segment_id"]
        word_start = float(w.get("word_start_sec",0) or 0)
        word_end   = float(w.get("word_end_sec",0) or 0)
        wav_path   = wav_index.get(sid,"")
        wi         = int(w.get("word_index",0))

        if wav_path and os.path.exists(wav_path):
            acous = extract_word_acoustics(wav_path, word_start, word_end)
        else:
            acous = {k:"" for k in ["f0_mean_hz","f0_min_hz","f0_max_hz",
                     "f0_range_hz","f0_std_hz","voiced_fraction",
                     "rms_energy","rms_energy_db","jitter","shimmer","hnr"]}
            acous["acoustic_status"] = "no_wav"

        pauses = pause_data.get(sid,{}).get(wi,(0.0,0.0))
        w["f0_mean_hz"]      = acous["f0_mean_hz"]
        w["f0_min_hz"]       = acous["f0_min_hz"]
        w["f0_max_hz"]       = acous["f0_max_hz"]
        w["f0_range_hz"]     = acous["f0_range_hz"]
        w["f0_std_hz"]       = acous["f0_std_hz"]
        w["voiced_fraction"] = acous["voiced_fraction"]
        w["rms_energy"]      = acous["rms_energy"]
        w["rms_energy_db"]   = acous["rms_energy_db"]
        w["jitter"]          = acous["jitter"]
        w["shimmer"]         = acous["shimmer"]
        w["hnr"]             = acous["hnr"]
        w["pause_before_ms"] = pauses[0]
        w["pause_after_ms"]  = pauses[1]
        w["speech_rate"]     = speech_rates.get(sid,"")
        w["acoustic_status"] = acous["acoustic_status"]

        st = acous["acoustic_status"]
        if st == "ok": ok += 1
        elif st == "too_short": short += 1
        else: err += 1

        if (i+1) % 200 == 0:
            log(f"[{i+1:5,}/{len(word_rows):,}] ok={ok} short={short} err={err}")

    log(f"Acoustic ok={ok} short={short} err={err}")
    return word_rows


# ── PHASE 5: Linguistic features ─────────────────────────────
def phase5_linguistic(word_rows, all_segments):
    banner("PHASE 5: Extracting Linguistic Features")

    # ── 5A: Simple derived ─────────────────────────────────────
    log("5A: Computing word_length_chars, log_duration, word_frequency ...")

    # Word frequency from corpus
    freq = Counter()
    for w in word_rows:
        wr = (w.get("word_resolved","") or "").strip().lower()
        if wr and wr not in ("<unk>","sp","sil","spn"):
            freq[wr] += 1
    log(f"  Unique word types : {len(freq):,}")

    for w in word_rows:
        wr  = (w.get("word_resolved","") or "").strip()
        dur = w.get("word_duration_ms","")

        w["word_length_chars"] = len(wr) if wr else 0

        try:
            d = float(dur)
            w["log_duration"] = round(math.log(d) if d > 0 else 0, 6)
        except:
            w["log_duration"] = ""

        wfreq = freq.get(wr.lower(), 1)
        w["word_frequency"] = wfreq
        w["log_word_freq"]  = round(math.log10(wfreq), 4)

    log("5A complete ✅")

    # ── 5B: Stanza features ────────────────────────────────────
    log("5B: Running Stanza dependency parsing ...")
    if not STANZA_OK:
        log("WARNING: Stanza not available — skipping")
        for w in word_rows:
            for col in ["word_stanza","pos_tag","dep_relation","head_word",
                        "dep_distance","lemma","xpos","feats",
                        "stanza_word_id","stanza_head_id","IC","SC"]:
                w[col] = ""
    else:
        nlp = stanza.Pipeline(
            lang="hi",
            processors="tokenize,pos,lemma,depparse",
            tokenize_pretokenized=True,
            verbose=False
        )
        log("  Stanza loaded ✅")

        # Group words by segment
        seg_words = defaultdict(list)
        for w in word_rows:
            seg_words[w["segment_id"]].append(w)

        # Map segment_id to clean_hindi_text
        seg_text = {s["segment_id"]: s.get("clean_hindi_text","")
                    for s in all_segments}

        total_segs = len(seg_words)
        for si, (seg_id, words) in enumerate(seg_words.items()):
            text = seg_text.get(seg_id,"").strip()
            if not text:
                for w in words:
                    for col in ["pos_tag","dep_relation","head_word",
                                "head_position","dep_distance","lemma",
                                "xpos","feats","stanza_word_id",
                                "stanza_head_id","IC","SC"]:
                        w[col] = ""
                continue
            try:
                tokens = text.split()
                doc    = nlp([tokens])
                sent   = doc.sentences[0]
                stanza_words = sent.words
                ic_list, sc_list = compute_ic_sc(stanza_words)

                for wi, w in enumerate(words):
                    if wi < len(stanza_words):
                        sw = stanza_words[wi]
                        hp = sw.head - 1
                        if sw.head == 0: hp = -1
                        hw = stanza_words[sw.head-1].text \
                             if sw.head > 0 else "ROOT"
                        dd = abs(sw.head - sw.id) if sw.head > 0 else 0
                        w["word_stanza"]     = sw.text
                        w["pos_tag"]         = sw.upos or ""
                        w["dep_relation"]    = sw.deprel or ""
                        w["head_word"]       = hw
                        w["dep_distance"]    = dd
                        w["lemma"]           = sw.lemma or sw.text
                        w["xpos"]            = sw.xpos or ""
                        w["feats"]           = sw.feats or ""
                        w["stanza_word_id"]  = sw.id
                        w["stanza_head_id"]  = sw.head
                        w["IC"]              = ic_list[wi]
                        w["SC"]              = sc_list[wi]
                    else:
                        for col in ["word_stanza","pos_tag","dep_relation",
                                    "head_word","dep_distance","lemma",
                                    "xpos","feats","stanza_word_id",
                                    "stanza_head_id","IC","SC"]:
                            w[col] = ""
            except Exception as e:
                for w in words:
                    for col in ["word_stanza","pos_tag","dep_relation",
                                "head_word","dep_distance","lemma",
                                "xpos","feats","stanza_word_id",
                                "stanza_head_id","IC","SC"]:
                        w[col] = ""

            if (si+1) % 50 == 0:
                log(f"  Stanza [{si+1:,}/{total_segs:,}]")

        log("5B Stanza complete ✅")

    # ── 5C: Surprisal from GPT-2 Hindi ────────────────────────
    log("5C: Computing forward/backward surprisal (GPT-2 Hindi) ...")
    if not GPT2_OK:
        log("WARNING: GPT-2 not available — skipping surprisal")
        for w in word_rows:
            w["forward_surprisal"]  = ""
            w["backward_surprisal"] = ""
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
        model     = GPT2LMHeadModel.from_pretrained(GPT2_MODEL)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        log(f"  GPT-2 Hindi loaded on {device} ✅")

        seg_words = defaultdict(list)
        for w in word_rows:
            seg_words[w["segment_id"]].append(w)

        seg_text = {s["segment_id"]: s.get("clean_hindi_text","")
                    for s in all_segments}

        total_segs = len(seg_words)
        for si, (seg_id, words) in enumerate(seg_words.items()):
            text = seg_text.get(seg_id,"").strip()
            if not text:
                for w in words:
                    w["forward_surprisal"]  = ""
                    w["backward_surprisal"] = ""
                continue

            tokens = text.split()

            # Forward surprisal
            try:
                fwd_surp = []
                for wi in range(len(tokens)):
                    context = " ".join(tokens[:wi]) if wi > 0 else ""
                    word    = tokens[wi]
                    if not context:
                        fwd_surp.append(0.0)
                        continue
                    inp = tokenizer(context, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out    = model(**inp)
                        logits = out.logits[0,-1,:]
                        probs  = torch.softmax(logits, dim=-1)
                    word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
                    if word_ids:
                        prob = probs[word_ids[0]].item()
                        surp = -math.log2(prob + 1e-10)
                    else:
                        surp = 0.0
                    fwd_surp.append(round(surp, 4))
            except:
                fwd_surp = [0.0] * len(tokens)

            # Backward surprisal (reverse sentence)
            try:
                rev_tokens = tokens[::-1]
                bwd_surp_rev = []
                for wi in range(len(rev_tokens)):
                    context = " ".join(rev_tokens[:wi]) if wi > 0 else ""
                    word    = rev_tokens[wi]
                    if not context:
                        bwd_surp_rev.append(0.0)
                        continue
                    inp = tokenizer(context, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out    = model(**inp)
                        logits = out.logits[0,-1,:]
                        probs  = torch.softmax(logits, dim=-1)
                    word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
                    if word_ids:
                        prob = probs[word_ids[0]].item()
                        surp = -math.log2(prob + 1e-10)
                    else:
                        surp = 0.0
                    bwd_surp_rev.append(round(surp, 4))
                bwd_surp = bwd_surp_rev[::-1]
            except:
                bwd_surp = [0.0] * len(tokens)

            # Assign to words
            for wi, w in enumerate(words):
                w["forward_surprisal"]  = fwd_surp[wi]  \
                                          if wi < len(fwd_surp) else ""
                w["backward_surprisal"] = bwd_surp[wi]  \
                                          if wi < len(bwd_surp) else ""

            if (si+1) % 20 == 0:
                log(f"  Surprisal [{si+1:,}/{total_segs:,}]")

        log("5C Surprisal complete ✅")

    return word_rows


# ── PHASE 6: Generate stanza_output.conllu ───────────────────
def phase6_conllu(word_rows, all_segments, conllu_path):
    banner("PHASE 6: Generating stanza_output.conllu")

    seg_text = {s["segment_id"]: s.get("clean_hindi_text","")
                for s in all_segments}
    seg_words = defaultdict(list)
    for w in word_rows:
        seg_words[w["segment_id"]].append(w)
    for sid in seg_words:
        seg_words[sid].sort(key=lambda r: int(r.get("word_index",0)))

    total_sents = 0
    total_words = 0

    with open(conllu_path, "w", encoding="utf-8") as f:
        for sid, words in seg_words.items():
            if not words: continue
            f.write(f"# segment_id = {sid}\n")
            text = seg_text.get(sid,"")
            f.write(f"# text = {text}\n")
            for w in words:
                token_id = int(w.get("word_index",0)) + 1
                form     = (w.get("word_resolved","") or "_") or "_"
                lemma    = (w.get("lemma","") or form) or form
                upos     = (w.get("pos_tag","") or "_") or "_"
                xpos     = (w.get("xpos","") or "_") or "_"
                feats    = (w.get("feats","") or "_") or "_"
                hp       = w.get("head_position","")
                try:
                    hp_int = int(hp)
                    head   = "0" if hp_int == -1 else str(hp_int+1)
                except:
                    head   = "_"
                deprel   = (w.get("dep_relation","") or "_") or "_"
                ic       = w.get("IC","")
                sc       = w.get("SC","")
                dd       = w.get("dep_distance","")
                misc_parts = []
                if ic != "": misc_parts.append(f"IC={ic}")
                if sc != "": misc_parts.append(f"SC={sc}")
                if dd != "": misc_parts.append(f"DepDist={dd}")
                misc = "|".join(misc_parts) if misc_parts else "_"
                line = "\t".join([str(token_id),form,lemma,
                                  upos,xpos,feats,head,deprel,"_",misc])
                f.write(line+"\n")
                total_words += 1
            f.write("\n")
            total_sents += 1

    log(f"Sentences written : {total_sents:,}")
    log(f"Words written     : {total_words:,}")
    log(f"Output            : {conllu_path}")


# ── Write CSVs ────────────────────────────────────────────────
def write_segments_csv(segments, out_path):
    if not segments: return
    with open(out_path,"w",encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(segments[0].keys()))
        writer.writeheader()
        writer.writerows(segments)
    sz = round(os.path.getsize(out_path)/1024/1024,1)
    log(f"segments.csv      : {len(segments):,} rows × {len(segments[0])} cols ({sz} MB)")

def write_word_csv(word_rows, out_path):
    if not word_rows: return

    # Explicit 128 column order:
    # 95 original + 12 stanza + 4 simple linguistic
    # + 2 surprisal + 15 acoustic
    FIELDNAMES = [
        # GROUP 1: Word + MFA (10 cols)
        "segment_id","word_index","word",
        "word_from_lab","word_resolved","word_is_resolved",
        "word_start_sec","word_end_sec","word_duration_ms","phone_count",
        # GROUP 2: Speaker (10 cols)
        "speaker_id_file","speaker_id_segment","speaker_id_note",
        "gender","age_group","area","state","district",
        "occupation","qualification",
        # GROUP 3: Segment metadata (14 cols)
        "scenario","scenario_corrected","task_name","domain",
        "quality_score","quality_score_v4","quality_acoustic_component",
        "quality_linguistic_deduction","dysfluency_density",
        "dataset_tier","is_acoustic_clean",
        "segment_duration_sec","segment_start_sec","segment_end_sec",
        # GROUP 4: Text content (6 cols)
        "verbatim_text","verbatim_word_count",
        "normalized_text","normalized_word_count",
        "clean_hindi_text","clean_hindi_word_count",
        # GROUP 5: Code switching (3 cols)
        "has_code_switch","code_switch_words","cs_word_count",
        # GROUP 6: Dysfluency (7 cols)
        "has_dysfluency","dysfluency_types",
        "verbatim_dysfluency_types","normalized_dysfluency_types",
        "dysfluency_count","verbatim_dysfluency_count",
        "normalized_dysfluency_count",
        # GROUP 7: Physiological (7 cols)
        "has_physiological","physiological_events",
        "verbatim_physiological_events","normalized_physiological_events",
        "physiological_count","verbatim_physiological_count",
        "normalized_physiological_count",
        # GROUP 8: Environmental noise (7 cols)
        "has_env_noise","env_noise_events",
        "verbatim_env_noise_events","normalized_env_noise_events",
        "env_noise_count","verbatim_env_noise_count",
        "normalized_env_noise_count",
        # GROUP 9: Quality annotations (7 cols)
        "has_quality_annotation","quality_annotations",
        "verbatim_quality_annotations","normalized_quality_annotations",
        "quality_annotation_count","verbatim_quality_annotation_count",
        "normalized_quality_annotation_count",
        # GROUP 10: Paralinguistic (7 cols)
        "has_paralinguistic","paralinguistic_events",
        "verbatim_paralinguistic_events","normalized_paralinguistic_events",
        "paralinguistic_count","verbatim_paralinguistic_count",
        "normalized_paralinguistic_count",
        # GROUP 11: Action (7 cols)
        "has_action","action_types",
        "verbatim_action_types","normalized_action_types",
        "action_count","verbatim_action_count","normalized_action_count",
        # GROUP 12: All annotations (4 cols)
        "all_annotations","all_annotation_count",
        "verbatim_all_annotation_count","normalized_all_annotation_count",
        # GROUP 13: Unknown annotations (6 cols)
        "unknown_annotations","verbatim_unknown_annotations",
        "normalized_unknown_annotations","unknown_annotation_count",
        "verbatim_unknown_annotation_count",
        "normalized_unknown_annotation_count",
        # ── NEW COLUMNS ────────────────────────────────────────
        # GROUP 14: Stanza features (12 cols)
        # Clean output — no word_position, no head_position (-1)
        # stanza_head_id = 0 for ROOT (CoNLL-U standard)
        "word_stanza","lemma",
        "stanza_word_id","stanza_head_id",
        "dep_relation","pos_tag","xpos","feats",
        "head_word","dep_distance","IC","SC",
        # GROUP 15: Simple linguistic (4 cols)
        "word_length_chars","log_duration",
        "word_frequency","log_word_freq",
        # GROUP 16: Surprisal from GPT-2 Hindi (2 cols)
        "forward_surprisal","backward_surprisal",
        # GROUP 17: Acoustic features from Parselmouth (15 cols)
        "f0_mean_hz","f0_min_hz","f0_max_hz",
        "f0_range_hz","f0_std_hz","voiced_fraction",
        "rms_energy","rms_energy_db",
        "jitter","shimmer","hnr",
        "pause_before_ms","pause_after_ms",
        "speech_rate","acoustic_status",
    ]

    with open(out_path,"w",encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(word_rows)
    sz = round(os.path.getsize(out_path)/1024/1024,1)
    log(f"word_features.csv : {len(word_rows):,} rows × {len(FIELDNAMES)} cols ({sz} MB) ✅")


# ── MAIN ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Indic Language Pipeline V2 — Full Feature Extraction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_file", help="Single JSON file")
    group.add_argument("--input_files", nargs="+", help="Multiple JSON files")
    group.add_argument("--input_dir", help="Folder with JSON + WAV files")
    parser.add_argument("--run_name", default="run",
                        help="Run name — creates separate output folder")
    parser.add_argument("--skip_surprisal", action="store_true",
                        help="Skip GPT-2 surprisal (faster)")
    args = parser.parse_args()

    # Collect JSON files
    json_files = []
    if args.input_file:
        p = os.path.abspath(args.input_file)
        if not os.path.exists(p):
            print(f"ERROR: {p} not found"); sys.exit(1)
        json_files = [p]
    elif args.input_files:
        json_files = [os.path.abspath(f) for f in args.input_files
                      if f.endswith(".json") and os.path.exists(f)]
    elif args.input_dir:
        d = os.path.abspath(args.input_dir)
        if not os.path.exists(d):
            print(f"ERROR: {d} not found"); sys.exit(1)
        json_files = sorted([os.path.join(d,f)
                             for f in os.listdir(d)
                             if f.endswith(".json")])
    if not json_files:
        print("ERROR: No JSON files found."); sys.exit(1)

    # Setup folders
    ROOT           = os.path.dirname(os.path.abspath(__file__))
    RUN_OUTPUT_DIR = os.path.join(ROOT,"outputs",args.run_name)
    MFA_SEG_DIR    = os.path.join(ROOT,"mfa_segments",args.run_name)
    MFA_ALN_DIR    = os.path.join(RUN_OUTPUT_DIR,"mfa_aligned")
    MFA_VAL_DIR    = os.path.join(RUN_OUTPUT_DIR,"mfa_validate_output")
    SEG_CSV        = os.path.join(RUN_OUTPUT_DIR,"segments.csv")
    WORD_CSV       = os.path.join(RUN_OUTPUT_DIR,"word_features.csv")
    CONLLU         = os.path.join(RUN_OUTPUT_DIR,"stanza_output.conllu")

    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MFA_SEG_DIR, exist_ok=True)

    start_time = time.time()

    print("\n"+"="*60)
    print("  INDIC LANGUAGE PIPELINE V2")
    print("  Full Feature Extraction (Acoustic + Linguistic)")
    print("  Student : Mayur Shriram Gaidhane (M22AIE248)")
    print("  Lab     : SAIDE Lab, IIT Jodhpur")
    print("="*60)
    print(f"\n  Run name     : {args.run_name}")
    print(f"  Input files  : {len(json_files)}")
    print(f"  Output folder: {RUN_OUTPUT_DIR}")

    # PHASE 1
    banner("PHASE 1: Parsing JSON + Cutting Audio + Preparing LAB Files")
    all_segments = []
    for idx, json_path in enumerate(json_files):
        print(f"\n  [{idx+1}/{len(json_files)}] {Path(json_path).name}")
        segs = phase1_parse_and_cut(json_path, MFA_SEG_DIR)
        all_segments.extend(segs)
        mfa_ready = sum(1 for s in segs if s["is_mfa_ready"])
        log(f"Segments: {len(segs)} total, {mfa_ready} MFA-ready")

    total_mfa = sum(1 for s in all_segments if s["is_mfa_ready"])
    log(f"Total segments: {len(all_segments):,} | MFA-ready: {total_mfa:,}")
    write_segments_csv(all_segments, SEG_CSV)

    # PHASE 2
    mfa_ok = phase2_mfa(MFA_SEG_DIR, MFA_ALN_DIR, MFA_VAL_DIR, RUN_OUTPUT_DIR)

    if not mfa_ok:
        log("Skipping word extraction — MFA produced no TextGrids.")
        elapsed = round(time.time()-start_time,1)
        print(f"\n  Time: {elapsed}s | segments.csv ready | word_features.csv skipped")
        return

    # PHASE 3
    word_rows = phase3_extract_words(all_segments, MFA_ALN_DIR)
    word_rows = phase3_resolve_unk(word_rows, MFA_SEG_DIR)

    # PHASE 4 — Acoustic
    word_rows = phase4_acoustic(word_rows, MFA_SEG_DIR, MFA_ALN_DIR)

    # PHASE 5 — Linguistic
    if not args.skip_surprisal:
        word_rows = phase5_linguistic(word_rows, all_segments)
    else:
        log("Surprisal skipped (--skip_surprisal flag)")
        for w in word_rows:
            w["word_length_chars"] = len((w.get("word_resolved","") or ""))
            try:
                d = float(w.get("word_duration_ms",0) or 0)
                w["log_duration"] = round(math.log(d) if d > 0 else 0, 6)
            except: w["log_duration"] = ""
            w["word_frequency"]     = 1
            w["log_word_freq"]      = 0
            w["forward_surprisal"]  = ""
            w["backward_surprisal"] = ""
            for col in ["pos_tag","dep_relation","head_word","head_position",
                        "dep_distance","lemma","xpos","feats",
                        "stanza_word_id","stanza_head_id","IC","SC"]:
                w[col] = ""

    # PHASE 6 — stanza_output.conllu
    phase6_conllu(word_rows, all_segments, CONLLU)

    # Write word_features.csv
    write_word_csv(word_rows, WORD_CSV)

    elapsed = round(time.time()-start_time,1)
    banner("PIPELINE V2 COMPLETE")
    print(f"\n  Run name     : {args.run_name}")
    print(f"  Time elapsed : {elapsed}s ({round(elapsed/60,1)} min)")
    print(f"\n  OUTPUT FILES:")
    print(f"  1. {SEG_CSV}")
    print(f"     → {len(all_segments):,} segments × {len(all_segments[0])} cols")
    print(f"  2. {WORD_CSV}")
    print(f"     → {len(word_rows):,} words × 128 cols")
    print(f"  3. {CONLLU}")
    print(f"     → stanza_output.conllu (CoNLL-U format)")
    print()


if __name__ == "__main__":
    main()