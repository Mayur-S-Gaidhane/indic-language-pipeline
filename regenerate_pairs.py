"""
Regenerate ground_truth_pairs with different context windows.
Reuses existing VTT + diarization files — no download or Whisper needed.
Takes ~10 seconds.

Usage:
    python regenerate_pairs.py --data_dir ground_truth_data

Output:
    ground_truth_pairs_clean.json      (context_window=3, original)
    ground_truth_pairs_ctx5_clean.json  (context_window=5)
    ground_truth_pairs_ctx8_clean.json  (context_window=8)
"""

import json
import os
import re
import sys
import argparse

# --- Reuse functions from youtube_ground_truth_pipeline.py ---

def parse_vtt(vtt_path):
    """Parse VTT subtitle file into segments."""
    segments = []
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by blank lines
    blocks = re.split(r'\n\s*\n', content)

    for block in blocks:
        lines = block.strip().split('\n')
        # Find timestamp line
        ts_line = None
        for line in lines:
            if '-->' in line:
                ts_line = line
                break
        if not ts_line:
            continue

        # Parse timestamps
        parts = ts_line.split('-->')
        start = timestamp_to_seconds(parts[0].strip())
        end = timestamp_to_seconds(parts[1].strip().split()[0])

        # Text = lines after timestamp
        text_lines = []
        found_ts = False
        for line in lines:
            if '-->' in line:
                found_ts = True
                continue
            if found_ts:
                # Remove VTT tags like <c>, </c>, etc.
                clean = re.sub(r'<[^>]+>', '', line).strip()
                if clean:
                    text_lines.append(clean)

        text = ' '.join(text_lines).strip()
        if text:
            segments.append({
                "start": start,
                "end": end,
                "text": text
            })

    # Merge consecutive segments with same text
    segments = merge_subtitle_segments(segments)
    print(f"  Parsed {len(segments)} subtitle segments from {os.path.basename(vtt_path)}")
    return segments


def timestamp_to_seconds(ts):
    """Convert HH:MM:SS.mmm or MM:SS.mmm to seconds."""
    parts = ts.replace(',', '.').split(':')
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def merge_subtitle_segments(segments):
    """Merge consecutive segments with identical text."""
    if not segments:
        return segments
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["text"] == merged[-1]["text"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def assign_speakers_to_subtitles(subtitle_segments, speaker_segments):
    """Assign speaker labels to subtitle segments using diarization."""
    labeled = []
    for seg in subtitle_segments:
        mid = (seg["start"] + seg["end"]) / 2
        best_speaker = "UNKNOWN"
        best_overlap = 0

        for spk_seg in speaker_segments:
            overlap_start = max(seg["start"], spk_seg["start"])
            overlap_end = min(seg["end"], spk_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk_seg["speaker"]

        labeled.append({
            "speaker": best_speaker,
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"]
        })
    return labeled


def _remove_overlap(existing, new_text):
    """Remove overlapping prefix from new_text that matches end of existing."""
    words_existing = existing.split()
    words_new = new_text.split()
    for overlap_len in range(min(len(words_existing), len(words_new)), 0, -1):
        if words_existing[-overlap_len:] == words_new[:overlap_len]:
            return ' '.join(words_new[overlap_len:])
    return new_text


def create_conversation_turns(labeled_segments):
    """Merge consecutive same-speaker segments into turns."""
    if not labeled_segments:
        return []

    turns = []
    current = labeled_segments[0].copy()

    for seg in labeled_segments[1:]:
        if seg["speaker"] == current["speaker"]:
            addition = _remove_overlap(current["text"], seg["text"])
            if addition:
                current["text"] += " " + addition
            current["end"] = seg["end"]
        else:
            turns.append(current)
            current = seg.copy()
    turns.append(current)

    print(f"  Created {len(turns)} conversation turns")
    return turns


def extract_pairs(turns, video_title, context_window=3):
    """Extract context-response pairs with given context window."""
    pairs = []
    for i in range(1, len(turns)):
        context_start = max(0, i - context_window)
        context_turns = turns[context_start:i]
        response_turn = turns[i]

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
    return pairs


def hindi_ratio(text):
    """Compute fraction of Hindi (Devanagari) characters."""
    if not text:
        return 0
    devanagari = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
    alpha = sum(1 for c in text if c.isalpha())
    return devanagari / alpha if alpha > 0 else 0


def filter_pairs_by_hindi_ratio(pairs, min_ratio=0.5):
    """Keep only pairs where both context and response are mostly Hindi."""
    filtered = []
    for pair in pairs:
        resp_ratio = hindi_ratio(pair["response"]["text"])
        ctx_ratios = [hindi_ratio(t["text"]) for t in pair["context"]]
        avg_ctx_ratio = sum(ctx_ratios) / len(ctx_ratios) if ctx_ratios else 0

        pair["response_hindi_ratio"] = round(resp_ratio, 3)
        pair["context_hindi_ratio"] = round(avg_ctx_ratio, 3)

        if resp_ratio >= min_ratio and avg_ctx_ratio >= min_ratio:
            filtered.append(pair)
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate ground truth pairs with different context windows")
    parser.add_argument("--data_dir", type=str, default="ground_truth_data",
                        help="Directory with VTT and diarization files")
    args = parser.parse_args()

    # Find VTT and diarization files
    vtt_file = os.path.join(args.data_dir, "video2_gadkari.hi.vtt")
    diar_file = os.path.join(args.data_dir, "video2_gadkari_diarization.json")

    if not os.path.exists(vtt_file):
        print(f"ERROR: VTT file not found: {vtt_file}")
        sys.exit(1)
    if not os.path.exists(diar_file):
        print(f"ERROR: Diarization file not found: {diar_file}")
        sys.exit(1)

    video_title = "Nitin Gadkari Interview - Megha Prasad"

    # Step 1: Parse subtitles
    print("Parsing subtitles...")
    subtitle_segments = parse_vtt(vtt_file)

    # Step 2: Load diarization
    print("Loading diarization...")
    with open(diar_file, "r") as f:
        speaker_segments = json.load(f)
    print(f"  Loaded {len(speaker_segments)} speaker segments")

    # Step 3: Merge
    print("Merging subtitles with speaker labels...")
    labeled = assign_speakers_to_subtitles(subtitle_segments, speaker_segments)
    turns = create_conversation_turns(labeled)

    # Step 4: Generate pairs for each context window
    for ctx_win in [3, 5, 8]:
        print(f"\n{'='*50}")
        print(f"Extracting pairs with context_window={ctx_win}")
        print(f"{'='*50}")

        pairs = extract_pairs(turns, video_title, context_window=ctx_win)
        print(f"  Raw pairs: {len(pairs)}")

        pairs = filter_pairs_by_hindi_ratio(pairs, min_ratio=0.5)
        print(f"  After Hindi filter: {len(pairs)}")

        # File naming
        if ctx_win == 3:
            filename = "ground_truth_pairs_clean.json"
        else:
            filename = f"ground_truth_pairs_ctx{ctx_win}_clean.json"

        output_path = os.path.join(args.data_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_pairs": len(pairs),
                "videos": {"video2_gadkari": video_title},
                "hindi_filter_threshold": 0.5,
                "context_window": ctx_win,
                "pairs": pairs
            }, f, ensure_ascii=False, indent=2)

        print(f"  Saved: {filename} ({len(pairs)} pairs)")

    print(f"\n{'='*50}")
    print("DONE! Files in", args.data_dir + ":")
    print("  ground_truth_pairs_clean.json       (context=3, original)")
    print("  ground_truth_pairs_ctx5_clean.json   (context=5)")
    print("  ground_truth_pairs_ctx8_clean.json   (context=8)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()