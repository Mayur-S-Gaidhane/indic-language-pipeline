"""
YouTube Ground-Truth Pipeline for MTP-3
========================================
Downloads Hindi YouTube videos, runs speaker diarization,
merges with subtitles, and extracts context-response pairs.

Usage:
    python youtube_ground_truth_pipeline.py

Output:
    ground_truth_pairs.json — context-response pairs ready for MTP-3

Requirements:
    pip install yt-dlp pyannote.audio pydub webvtt-py torch torchaudio

Note: pyannote.audio requires a HuggingFace token with access to:
    - pyannote/speaker-diarization-3.1
    - pyannote/segmentation-3.0
    Accept the user agreements at:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/segmentation-3.0
    Then: huggingface-cli login
"""

import subprocess
import json
import re
import os
import sys

# ============================================================
# STEP 0: CONFIG
# ============================================================

VIDEOS = {
    "video2_gadkari": {
        "url": "https://www.youtube.com/live/MISGh39Faas",
        "title": "Nitin Gadkari Interview - Megha Prasad"
    },
    # "video4_gaurav": {
    #     "url": "https://youtu.be/vAyEJSGi_n0",
    #     "title": "Gaurav Kumar Tripathi UPSC Mock Interview"
    # }
}

OUTPUT_DIR = "ground_truth_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: DOWNLOAD AUDIO + SUBTITLES
# ============================================================

def download_video(video_id, url):
    """Download audio (wav) and Hindi auto-subtitles (vtt).
    Skips download if files already exist."""
    output_path = os.path.join(OUTPUT_DIR, video_id)

    wav_file = f"{output_path}.wav"
    # Check for existing vtt file
    vtt_file = None
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith(video_id) and ".hi." in f and f.endswith(".vtt"):
                vtt_file = os.path.join(OUTPUT_DIR, f)
                break

    # Skip download if both files already exist
    if os.path.exists(wav_file) and vtt_file and os.path.exists(vtt_file):
        print(f"\n{'='*60}")
        print(f"SKIPPING DOWNLOAD (files already exist): {video_id}")
        print(f"  WAV: {wav_file}")
        print(f"  VTT: {vtt_file}")
        print(f"{'='*60}")
        return wav_file, vtt_file

    print(f"\n{'='*60}")
    print(f"Downloading: {video_id}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    # Download audio as wav
    if not os.path.exists(wav_file):
        cmd_audio = [
            "yt-dlp", "-x", "--audio-format", "wav",
            "-o", f"{output_path}.%(ext)s",
            url
        ]
        print(f"\n[1/2] Downloading audio...")
        subprocess.run(cmd_audio, check=True)
    else:
        print(f"[1/2] Audio already exists: {wav_file}")

    # Download Hindi auto-subtitles
    if not vtt_file:
        cmd_subs = [
            "yt-dlp", "--write-auto-sub", "--sub-lang", "hi",
            "--skip-download",
            "-o", f"{output_path}.%(ext)s",
            url
        ]
        print(f"[2/2] Downloading Hindi subtitles...")
        subprocess.run(cmd_subs, check=True)

        # Re-check for vtt file
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith(video_id) and ".hi." in f and f.endswith(".vtt"):
                vtt_file = os.path.join(OUTPUT_DIR, f)
                break
    else:
        print(f"[2/2] Subtitles already exist: {vtt_file}")

    if not os.path.exists(wav_file):
        print(f"WARNING: wav file not found at {wav_file}")
        print(f"Files in {OUTPUT_DIR}: {os.listdir(OUTPUT_DIR)}")
    if not vtt_file:
        print(f"WARNING: vtt subtitle file not found for {video_id}")
        print(f"Files in {OUTPUT_DIR}: {os.listdir(OUTPUT_DIR)}")

    return wav_file, vtt_file


# ============================================================
# STEP 2: PARSE VTT SUBTITLES
# ============================================================

def parse_vtt(vtt_path):
    """Parse .vtt file into list of {start, end, text} segments.
    Handles YouTube auto-generated VTT format with <c> tags."""
    segments = []

    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    timestamp_pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"

    i = 0
    while i < len(lines):
        match = re.match(timestamp_pattern, lines[i].strip())
        if match:
            start_str, end_str = match.group(1), match.group(2)
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(timestamp_pattern, lines[i].strip()):
                # Clean HTML/VTT tags like <c>, </c>, <00:00:10.480>
                clean = re.sub(r'<[^>]+>', '', lines[i]).strip()
                if clean:
                    text_lines.append(clean)
                i += 1
            text = ' '.join(text_lines)
            if text:
                start_sec = timestamp_to_seconds(start_str)
                end_sec = timestamp_to_seconds(end_str)
                segments.append({
                    "start": start_sec,
                    "end": end_sec,
                    "text": text
                })
        else:
            i += 1

    # Merge consecutive segments with same/overlapping text
    merged = merge_subtitle_segments(segments)
    print(f"  Parsed {len(segments)} raw segments -> {len(merged)} merged segments")
    return merged


def timestamp_to_seconds(ts):
    """Convert HH:MM:SS.mmm to seconds"""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def merge_subtitle_segments(segments):
    """Merge overlapping/duplicate subtitle segments from auto-captions"""
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        # If texts overlap significantly or are very close in time
        if seg["start"] - prev["end"] < 0.5 and seg["text"] in prev["text"]:
            # Extend end time, text already included
            prev["end"] = max(prev["end"], seg["end"])
        elif seg["start"] - prev["end"] < 0.5 and prev["text"] in seg["text"]:
            # New text is superset
            prev["text"] = seg["text"]
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())

    return merged


# ============================================================
# STEP 3: SPEAKER DIARIZATION
# ============================================================

def run_diarization(wav_path):
    """Run pyannote speaker diarization on audio file.
    Caches results to a JSON file to avoid re-running."""

    # Check for cached diarization
    cache_path = wav_path.replace(".wav", "_diarization.json")
    if os.path.exists(cache_path):
        print(f"\n  Loading cached diarization from {cache_path}")
        with open(cache_path, "r") as f:
            speaker_segments = json.load(f)
        print(f"  Found {len(speaker_segments)} speaker segments (cached)")
        speakers = set(s["speaker"] for s in speaker_segments)
        for sp in speakers:
            count = sum(1 for s in speaker_segments if s["speaker"] == sp)
            total_dur = sum(s["end"] - s["start"] for s in speaker_segments if s["speaker"] == sp)
            print(f"    {sp}: {count} segments, {total_dur:.1f}s total")
        return speaker_segments

    print(f"\n  Running speaker diarization on {wav_path}...")
    print(f"  (This may take several minutes depending on audio length)")

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("ERROR: pyannote.audio not installed.")
        print("Run: pip install pyannote.audio")
        sys.exit(1)

    # Load pipeline (requires HuggingFace token)
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )
    except Exception as e:
        print(f"ERROR loading pyannote pipeline: {e}")
        print("\nMake sure you have:")
        print("1. Accepted user agreements at:")
        print("   https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   https://huggingface.co/pyannote/segmentation-3.0")
        print("2. Logged in: huggingface-cli login")
        sys.exit(1)

    # Load audio manually to avoid torchcodec dependency
    import torch
    import torchaudio
    waveform, sample_rate = torchaudio.load(wav_path)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    # Run diarization (specify 2 speakers since we know it's an interview)
    diarization = pipeline(audio_input, num_speakers=2)

    # Convert to list of segments
    # Handle both old Annotation API (.itertracks) and new DiarizeOutput API
    speaker_segments = []
    if hasattr(diarization, 'itertracks'):
        # pyannote <= 3.1: returns Annotation object with .itertracks()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
    elif hasattr(diarization, 'speaker_diarization'):
        # pyannote >= 3.3 (community-1, etc.): returns DiarizeOutput
        # .speaker_diarization yields (turn, speaker) 2-tuples
        for turn, speaker in diarization.speaker_diarization:
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
    else:
        # Unknown API — print available attrs for debugging and exit
        attrs = [a for a in dir(diarization) if not a.startswith('_')]
        print(f"  ERROR: Unknown diarization output type: {type(diarization)}")
        print(f"  Available attributes: {attrs}")
        print(f"  Try: pip install pyannote.audio==3.1.1")
        sys.exit(1)

    print(f"  Found {len(speaker_segments)} speaker segments")

    # Cache diarization results
    with open(cache_path, "w") as f:
        json.dump(speaker_segments, f, indent=2)
    print(f"  Cached diarization to {cache_path}")

    # Count per speaker
    speakers = set(s["speaker"] for s in speaker_segments)
    for sp in speakers:
        count = sum(1 for s in speaker_segments if s["speaker"] == sp)
        total_dur = sum(s["end"] - s["start"] for s in speaker_segments if s["speaker"] == sp)
        print(f"    {sp}: {count} segments, {total_dur:.1f}s total")

    return speaker_segments


# ============================================================
# STEP 4: MERGE SUBTITLES + SPEAKER LABELS
# ============================================================

def assign_speakers_to_subtitles(subtitle_segments, speaker_segments):
    """Assign speaker labels to subtitle segments based on time overlap"""
    labeled = []

    for sub in subtitle_segments:
        sub_mid = (sub["start"] + sub["end"]) / 2

        # Find which speaker segment contains the midpoint of this subtitle
        best_speaker = "UNKNOWN"
        best_overlap = 0

        for spk in speaker_segments:
            # Calculate overlap
            overlap_start = max(sub["start"], spk["start"])
            overlap_end = min(sub["end"], spk["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk["speaker"]

        labeled.append({
            "start": sub["start"],
            "end": sub["end"],
            "text": sub["text"],
            "speaker": best_speaker
        })

    return labeled


def _remove_overlap(existing, new_text):
    """Remove overlapping prefix of new_text that duplicates the end of existing.
    WhisperX segments often have partial overlaps like:
      existing = '...आपको एक बात बता दूं'
      new_text = 'आपको एक बात बता दूं बोलो'
    Result:  '...आपको एक बात बता दूं बोलो'  (overlap removed, new suffix kept)
    """
    if not existing or not new_text:
        return new_text
    # Try progressively longer suffixes of existing text
    # to find overlap with prefix of new_text
    words_existing = existing.split()
    words_new = new_text.split()
    best_overlap = 0
    # Check if last N words of existing match first N words of new_text
    max_check = min(len(words_existing), len(words_new))
    for n in range(1, max_check + 1):
        if words_existing[-n:] == words_new[:n]:
            best_overlap = n
    if best_overlap > 0:
        # Return only the non-overlapping suffix from new_text
        remainder = words_new[best_overlap:]
        return " ".join(remainder) if remainder else ""
    return new_text


def create_conversation_turns(labeled_segments):
    """Group consecutive same-speaker segments into conversation turns.
    Deduplicates overlapping WhisperX text at segment boundaries."""
    if not labeled_segments:
        return []

    turns = []
    current_turn = {
        "speaker": labeled_segments[0]["speaker"],
        "start": labeled_segments[0]["start"],
        "end": labeled_segments[0]["end"],
        "text": labeled_segments[0]["text"]
    }

    for seg in labeled_segments[1:]:
        if seg["speaker"] == current_turn["speaker"]:
            # Same speaker continues — deduplicate overlap
            current_turn["end"] = seg["end"]
            new_part = _remove_overlap(current_turn["text"], seg["text"])
            if new_part:
                current_turn["text"] += " " + new_part
        else:
            # Speaker changed -> new turn
            turns.append(current_turn)
            current_turn = {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }

    turns.append(current_turn)

    print(f"  Created {len(turns)} conversation turns")
    return turns


# ============================================================
# STEP 5: EXTRACT CONTEXT-RESPONSE PAIRS
# ============================================================

def extract_pairs(turns, video_title, context_window=3):
    """
    Extract context-response pairs for MTP-3.

    For each turn (as response), the context is the previous
    `context_window` turns.

    Returns list of:
    {
        "context": [list of previous turns],
        "response": "the actual human response",
        "response_speaker": "SPEAKER_XX",
        "video": "source video title",
        "response_start": start_time,
        "response_end": end_time
    }
    """
    pairs = []

    for i in range(1, len(turns)):
        # Context = previous turns (up to context_window)
        context_start = max(0, i - context_window)
        context_turns = turns[context_start:i]

        response_turn = turns[i]

        # Skip very short responses (likely backchannels like "हां", "जी")
        if len(response_turn["text"].split()) < 3:
            continue

        pair = {
            "pair_id": f"{video_title[:20]}_{i}",
            "context": [
                {
                    "speaker": t["speaker"],
                    "text": t["text"],
                    "start": round(t["start"], 2),
                    "end": round(t["end"], 2)
                }
                for t in context_turns
            ],
            "response": {
                "speaker": response_turn["speaker"],
                "text": response_turn["text"],
                "start": round(response_turn["start"], 2),
                "end": round(response_turn["end"], 2)
            },
            "video": video_title
        }
        pairs.append(pair)

    print(f"  Extracted {len(pairs)} context-response pairs")
    return pairs


# ============================================================
# STEP 6: COMPUTE HINDI RATIO (filter Hinglish)
# ============================================================

def hindi_ratio(text):
    """Calculate ratio of Devanagari characters in text"""
    if not text:
        return 0
    # Devanagari Unicode range: 0900-097F
    devanagari_chars = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars == 0:
        return 0
    return devanagari_chars / alpha_chars


def filter_pairs_by_hindi_ratio(pairs, min_ratio=0.5):
    """Keep only pairs where both context and response are mostly Hindi"""
    filtered = []
    for pair in pairs:
        # Check response Hindi ratio
        resp_ratio = hindi_ratio(pair["response"]["text"])

        # Check average context Hindi ratio
        ctx_ratios = [hindi_ratio(t["text"]) for t in pair["context"]]
        avg_ctx_ratio = sum(ctx_ratios) / len(ctx_ratios) if ctx_ratios else 0

        pair["response_hindi_ratio"] = round(resp_ratio, 3)
        pair["context_hindi_ratio"] = round(avg_ctx_ratio, 3)

        if resp_ratio >= min_ratio and avg_ctx_ratio >= min_ratio:
            filtered.append(pair)

    print(f"  Hindi filter (>={min_ratio}): {len(filtered)}/{len(pairs)} pairs kept")
    return filtered


# ============================================================
# MAIN
# ============================================================

def main():
    all_pairs = []

    for video_id, info in VIDEOS.items():
        print(f"\n{'#'*60}")
        print(f"# Processing: {info['title']}")
        print(f"{'#'*60}")

        # Step 1: Download
        wav_path, vtt_path = download_video(video_id, info["url"])

        if not vtt_path or not os.path.exists(vtt_path):
            print(f"SKIPPING {video_id}: no subtitle file found")
            continue

        # Step 2: Parse subtitles
        print(f"\n  Parsing subtitles...")
        subtitle_segments = parse_vtt(vtt_path)

        # Step 3: Speaker diarization
        speaker_segments = run_diarization(wav_path)

        # Step 4: Merge
        print(f"\n  Merging subtitles with speaker labels...")
        labeled = assign_speakers_to_subtitles(subtitle_segments, speaker_segments)
        turns = create_conversation_turns(labeled)

        # Print first few turns as sanity check
        print(f"\n  First 5 turns:")
        for t in turns[:5]:
            preview = t["text"][:80] + "..." if len(t["text"]) > 80 else t["text"]
            print(f"    [{t['speaker']}] ({t['start']:.1f}s-{t['end']:.1f}s): {preview}")

        # Step 5: Extract pairs
        print(f"\n  Extracting context-response pairs...")
        pairs = extract_pairs(turns, info["title"])

        # Step 6: Filter by Hindi ratio
        print(f"\n  Filtering by Hindi content ratio...")
        pairs = filter_pairs_by_hindi_ratio(pairs, min_ratio=0.5)

        all_pairs.extend(pairs)

    # Save output
    output_file = os.path.join(OUTPUT_DIR, "ground_truth_pairs.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": len(all_pairs),
            "videos": {vid: info["title"] for vid, info in VIDEOS.items()},
            "hindi_filter_threshold": 0.5,
            "context_window": 3,
            "pairs": all_pairs
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Saved {len(all_pairs)} pairs to {output_file}")
    print(f"{'='*60}")
    print(f"\nUpload {output_file} to the Claude chat for MTP-3 scoring.")


if __name__ == "__main__":
    main()