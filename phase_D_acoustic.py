#!/usr/bin/env python3
"""
phase_D_acoustic.py — Phase D: Acoustic Features (Parselmouth)

Uses MTP-2 approach:
  Load WAV ONCE per segment (not per word)
  Use snd.extract_part() for each word time slice
  28,302 WAV loads vs 614,351 (21x fewer file reads)

Extracts 15 acoustic features per word:
  f0_mean_hz, f0_min_hz, f0_max_hz, f0_range_hz, f0_std_hz
  voiced_fraction
  rms_energy, rms_energy_db
  jitter, shimmer, hnr
  pause_before_ms, pause_after_ms
  speech_rate, acoustic_status

INPUT : outputs/results/word_features_v4.csv (614,351 × 95)
        mfa_segments/     (WAV files)
        outputs/mfa_aligned/ (TextGrid files for pauses)
OUTPUT: outputs/results/features_D_acoustic.csv (614,351 × 17)

Time  : ~1-2 hours (vs 5 days with per-word loading)
Resume: python phase_D_acoustic.py --resume

Student   : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os
import csv
import math
import time
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

ROOT       = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(ROOT, "outputs", "results")
WORD_CSV   = os.path.join(RESULTS, "word_features_v4.csv")
MFA_SEG    = os.path.join(ROOT, "mfa_segments")
MFA_ALN    = os.path.join(ROOT, "outputs", "mfa_aligned")
OUTPUT_CSV = os.path.join(RESULTS, "features_D_acoustic.csv")
PROGRESS_F = os.path.join(RESULTS, "phase_D_progress.txt")

# Praat settings (same as MTP-2)
F0_MIN_HZ    = 75
F0_MAX_HZ    = 500
TIME_STEP    = 0.01
MIN_DUR_SEC  = 0.02   # 20ms minimum for any features
MIN_DUR_JITTER = 0.05 # 50ms minimum for jitter/shimmer/HNR
CHECKPOINT_N = 1000   # save progress every N segments

FIELDNAMES = [
    "segment_id", "word_index",
    "f0_mean_hz", "f0_min_hz", "f0_max_hz",
    "f0_range_hz", "f0_std_hz", "voiced_fraction",
    "rms_energy", "rms_energy_db",
    "jitter", "shimmer", "hnr",
    "pause_before_ms", "pause_after_ms",
    "speech_rate", "acoustic_status",
]

try:
    import numpy as np
    import parselmouth
    from parselmouth.praat import call
    PARSEL_OK = True
except ImportError:
    PARSEL_OK = False
    print("ERROR: parselmouth not installed")


def extract_word_acoustics(snd, start_sec, end_sec):
    """
    Extract acoustic features for one word.
    snd = parselmouth.Sound of FULL SEGMENT (loaded once).
    Uses snd.extract_part() to slice word audio.
    Same as MTP-2 extract_acoustic_features_v4.py logic.
    """
    empty = {k: "" for k in [
        "f0_mean_hz", "f0_min_hz", "f0_max_hz",
        "f0_range_hz", "f0_std_hz", "voiced_fraction",
        "rms_energy", "rms_energy_db",
        "jitter", "shimmer", "hnr"]}
    empty["acoustic_status"] = "error"

    duration = end_sec - start_sec
    if duration < MIN_DUR_SEC:
        empty["acoustic_status"] = "too_short"
        return empty

    try:
        # Slice word audio from segment Sound object
        word_snd = snd.extract_part(
            from_time=start_sec,
            to_time=end_sec,
            preserve_times=False
        )

        result = dict(empty)

        # RMS Energy
        try:
            rms = word_snd.get_rms()
            if rms and rms > 0:
                result["rms_energy"]    = round(float(rms), 6)
                result["rms_energy_db"] = round(
                    float(10 * np.log10(rms**2 + 1e-10)), 3)
        except: pass

        # F0 / Pitch
        try:
            pitch  = word_snd.to_pitch(
                TIME_STEP, F0_MIN_HZ, F0_MAX_HZ)
            f0_vals = [
                pitch.get_value_at_time(t)
                for t in np.arange(0, duration, TIME_STEP)
                if pitch.get_value_at_time(t) and
                   not np.isnan(pitch.get_value_at_time(t))
            ]
            voiced = [v for v in f0_vals if v > 0]
            if voiced:
                result["f0_mean_hz"]      = round(float(np.mean(voiced)), 2)
                result["f0_min_hz"]       = round(float(np.min(voiced)), 2)
                result["f0_max_hz"]       = round(float(np.max(voiced)), 2)
                result["f0_range_hz"]     = round(float(
                    np.max(voiced) - np.min(voiced)), 2)
                result["f0_std_hz"]       = round(float(np.std(voiced)), 2)
                result["voiced_fraction"] = round(
                    len(voiced) / max(1, len(f0_vals)), 3)
            else:
                for k in ["f0_mean_hz","f0_min_hz","f0_max_hz",
                          "f0_range_hz","f0_std_hz","voiced_fraction"]:
                    result[k] = 0
        except: pass

        # Jitter, Shimmer, HNR (needs >= 50ms)
        if duration >= MIN_DUR_JITTER:
            try:
                pp = call(word_snd,
                    "To PointProcess (periodic, cc)",
                    F0_MIN_HZ, F0_MAX_HZ)
                j = call(pp, "Get jitter (local)",
                         0, 0, 0.0001, 0.02, 1.3)
                if j and not np.isnan(j):
                    result["jitter"] = round(float(j), 6)
            except: pass
            try:
                pp = call(word_snd,
                    "To PointProcess (periodic, cc)",
                    F0_MIN_HZ, F0_MAX_HZ)
                sh = call([word_snd, pp],
                    "Get shimmer (local)",
                    0, 0, 0.0001, 0.02, 1.3, 1.6)
                if sh and not np.isnan(sh):
                    result["shimmer"] = round(float(sh), 6)
            except: pass
            try:
                harm = call(word_snd,
                    "To Harmonicity (cc)",
                    0.01, F0_MIN_HZ, 0.1, 1.0)
                hnr = call(harm, "Get mean", 0, 0)
                if hnr and not np.isnan(hnr):
                    result["hnr"] = round(float(hnr), 3)
            except: pass

        result["acoustic_status"] = (
            "ok_short" if duration < MIN_DUR_JITTER else "ok")
        return result

    except Exception as e:
        empty["acoustic_status"] = "error"
        return empty


def read_pauses(tg_path):
    """Read pause durations from TextGrid silence intervals."""
    words_tier   = []
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
                cur_text = ls.split("=", 1)[1].strip().strip('"')
                if cur_xmin is not None and cur_xmax is not None:
                    words_tier.append((cur_xmin, cur_xmax, cur_text))
                cur_xmin = cur_xmax = cur_text = None

    word_entries = [(x, y, t) for x, y, t in words_tier
                   if t and t not in ("", "sp", "sil", "SIL")]
    pause_map = {}
    for wi, (w_xmin, w_xmax, _) in enumerate(word_entries):
        pb = 0.0
        for xmin, xmax, text in words_tier:
            if xmax <= w_xmin and (not text or
                    text in ("", "sp", "sil")):
                if w_xmin - xmax < 0.001:
                    pb = (xmax - xmin) * 1000
        pa = 0.0
        for xmin, xmax, text in words_tier:
            if xmin >= w_xmax and (not text or
                    text in ("", "sp", "sil")):
                if xmin - w_xmax < 0.001:
                    pa = (xmax - xmin) * 1000
                    break
        pause_map[wi] = (round(pb, 1), round(pa, 1))
    return pause_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
        help="Resume from last checkpoint")
    args = parser.parse_args()

    print("="*60)
    print("  PHASE D: Acoustic Features (Parselmouth)")
    print("  Method: MTP-2 approach — load WAV once per segment")
    print("  f0, rms, jitter, shimmer, hnr, pauses, speech_rate")
    print("="*60)

    if not PARSEL_OK:
        print("ERROR: parselmouth not installed")
        return

    start = time.time()

    # Resume support
    start_seg = 0
    if args.resume and os.path.exists(PROGRESS_F):
        with open(PROGRESS_F) as f:
            start_seg = int(f.read().strip())
        print(f"\n  Resuming from segment {start_seg:,}")

    # Index WAV files: seg_id → wav_path
    print(f"\n  Indexing WAV files ...")
    wav_index = {}
    for wp in Path(MFA_SEG).rglob("*.wav"):
        wav_index[wp.stem] = str(wp)
    print(f"  WAV files indexed: {len(wav_index):,}")

    # Index TextGrid pauses
    print(f"  Loading TextGrid pauses ...")
    tg_files   = list(Path(MFA_ALN).rglob("*.TextGrid"))
    pause_data = {}
    for tg in tg_files:
        try: pause_data[tg.stem] = read_pauses(str(tg))
        except: pass
    print(f"  TextGrid files   : {len(pause_data):,}")

    # Load word rows grouped by segment
    print(f"\n  Loading word_features_v4.csv ...")
    seg_words  = defaultdict(list)  # seg_id → [word_row_dicts]
    seg_order  = []
    total_words = 0
    with open(WORD_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row["segment_id"]
            if sid not in seg_words:
                seg_order.append(sid)
            seg_words[sid].append({
                "segment_id"          : row["segment_id"],
                "word_index"          : row["word_index"],
                "word_start_sec"      : row.get("word_start_sec",""),
                "word_end_sec"        : row.get("word_end_sec",""),
                "segment_duration_sec": row.get("segment_duration_sec",""),
            })
            total_words += 1
    print(f"  Words loaded     : {total_words:,}")
    print(f"  Segments         : {len(seg_order):,}")

    # Speech rate per segment
    speech_rates = {}
    for sid, words in seg_words.items():
        try:
            dur = float(words[0]["segment_duration_sec"] or 0)
            speech_rates[sid] = round(
                len(words) / dur, 3) if dur > 0 else 0
        except:
            speech_rates[sid] = 0

    # Process segment by segment
    total_segs = len(seg_order)
    ok=0; short_c=0; err=0; no_wav=0
    done_words = 0

    mode = "a" if start_seg > 0 else "w"

    with open(OUTPUT_CSV, mode, encoding="utf-8-sig",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if start_seg == 0:
            writer.writeheader()

        for si, seg_id in enumerate(seg_order):
            if si < start_seg:
                continue

            words        = seg_words[seg_id]
            wav_path     = wav_index.get(seg_id, "")
            seg_pauses   = pause_data.get(seg_id, {})
            speech_rate  = speech_rates.get(seg_id, "")

            # Load segment WAV ONCE
            snd = None
            if wav_path and os.path.exists(wav_path):
                try:
                    snd = parselmouth.Sound(wav_path)
                except:
                    snd = None

            for wi, word in enumerate(words):
                try:
                    ws = float(word["word_start_sec"] or 0)
                    we = float(word["word_end_sec"] or 0)
                except:
                    ws = we = 0

                if snd is not None:
                    acous = extract_word_acoustics(snd, ws, we)
                else:
                    acous = {k: "" for k in [
                        "f0_mean_hz","f0_min_hz","f0_max_hz",
                        "f0_range_hz","f0_std_hz","voiced_fraction",
                        "rms_energy","rms_energy_db",
                        "jitter","shimmer","hnr"]}
                    acous["acoustic_status"] = "no_wav"

                pauses = seg_pauses.get(wi, (0.0, 0.0))
                st     = acous["acoustic_status"]
                if st == "ok":       ok      += 1
                elif st == "too_short": short_c += 1
                elif st == "no_wav":    no_wav  += 1
                else:                   err     += 1

                writer.writerow({
                    "segment_id"     : seg_id,
                    "word_index"     : word["word_index"],
                    "f0_mean_hz"     : acous["f0_mean_hz"],
                    "f0_min_hz"      : acous["f0_min_hz"],
                    "f0_max_hz"      : acous["f0_max_hz"],
                    "f0_range_hz"    : acous["f0_range_hz"],
                    "f0_std_hz"      : acous["f0_std_hz"],
                    "voiced_fraction": acous["voiced_fraction"],
                    "rms_energy"     : acous["rms_energy"],
                    "rms_energy_db"  : acous["rms_energy_db"],
                    "jitter"         : acous["jitter"],
                    "shimmer"        : acous["shimmer"],
                    "hnr"            : acous["hnr"],
                    "pause_before_ms": pauses[0],
                    "pause_after_ms" : pauses[1],
                    "speech_rate"    : speech_rate,
                    "acoustic_status": st,
                })
                done_words += 1

            # Checkpoint every N segments
            if (si + 1) % CHECKPOINT_N == 0:
                with open(PROGRESS_F, "w") as pf:
                    pf.write(str(si + 1))
                pct     = round(100*(si+1)/total_segs, 1)
                elapsed = round(time.time()-start, 0)
                rate    = (si+1-start_seg) / max(elapsed, 1)
                eta     = round((total_segs-si-1)/max(rate,0.01),0)
                print(f"  [{si+1:,}/{total_segs:,}] {pct}%"
                      f" | ok={ok} short={short_c}"
                      f" no_wav={no_wav} err={err}"
                      f" | elapsed={int(elapsed)}s"
                      f" ETA={int(eta)}s"
                      f" | resume: --resume {si+1}")
                f.flush()

    # Cleanup progress file
    if os.path.exists(PROGRESS_F):
        os.remove(PROGRESS_F)

    sz      = round(os.path.getsize(OUTPUT_CSV)/1024/1024, 1)
    elapsed = round(time.time()-start, 1)

    print()
    print("="*60)
    print("  PHASE D COMPLETE")
    print("="*60)
    print(f"  Words processed  : {done_words:,}")
    print(f"  Acoustic ok      : {ok:,}")
    print(f"  Too short        : {short_c:,}")
    print(f"  No WAV           : {no_wav:,}")
    print(f"  Errors           : {err:,}")
    print(f"  Columns          : {len(FIELDNAMES)}")
    print(f"  Size             : {sz} MB")
    print(f"  Time             : {elapsed}s ({round(elapsed/3600,1)}h)")
    print(f"  Output           : {OUTPUT_CSV}")
    print("="*60)


if __name__ == "__main__":
    main()