"""
Expand context window in ground_truth_pairs_clean.json from 3 to 5 (or 8).
Reconstructs full turn sequence from overlapping pairs — no re-parsing needed.

ONLY READS the original file, creates NEW files. No existing files modified.

Usage:
    python expand_context.py --input ground_truth_data/ground_truth_pairs_clean.json

Output:
    ground_truth_data/ground_truth_pairs_ctx5_clean.json
    ground_truth_data/ground_truth_pairs_ctx8_clean.json
"""

import json
import os
import argparse


def expand_context(pairs, all_turns, turn_index, context_window):
    """Re-create pairs with a larger context window."""
    new_pairs = []
    for p in pairs:
        resp = p['response']
        resp_key = (resp['start'], resp['end'])
        resp_idx = turn_index.get(resp_key)

        if resp_idx is None:
            # Fallback: keep original pair unchanged
            new_pairs.append(p)
            continue

        # Get up to context_window turns before this response
        ctx_start = max(0, resp_idx - context_window)
        ctx_turns = all_turns[ctx_start:resp_idx]

        new_pair = {
            'pair_id': p['pair_id'],
            'context': [
                {
                    'speaker': t['speaker'],
                    'text': t['text'],
                    'start': t['start'],
                    'end': t['end']
                }
                for t in ctx_turns
            ],
            'response': resp,
            'video': p['video'],
            'response_hindi_ratio': p.get('response_hindi_ratio', 0),
            'context_hindi_ratio': p.get('context_hindi_ratio', 0),
        }
        new_pairs.append(new_pair)

    return new_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Expand context window in ground truth pairs (read-only on original)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to ground_truth_pairs_clean.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        return

    # Load original data
    print(f"Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data['pairs']
    print(f"  Original: {len(pairs)} pairs, context_window={data.get('context_window', 3)}")

    # Step 1: Reconstruct full turn sequence from overlapping pairs
    print("Reconstructing full turn sequence...")
    turns_by_pos = {}
    for p in pairs:
        for t in p['context']:
            key = (t['start'], t['end'])
            if key not in turns_by_pos:
                turns_by_pos[key] = t
        r = p['response']
        key = (r['start'], r['end'])
        if key not in turns_by_pos:
            turns_by_pos[key] = r

    all_turns = sorted(turns_by_pos.values(), key=lambda t: t['start'])
    turn_index = {(t['start'], t['end']): i for i, t in enumerate(all_turns)}
    print(f"  Reconstructed {len(all_turns)} unique turns")

    # Step 2: Generate expanded versions
    output_dir = os.path.dirname(args.input)

    for ctx_win in [5, 8]:
        new_pairs = expand_context(pairs, all_turns, turn_index, ctx_win)

        # Verify integrity
        assert len(new_pairs) == len(pairs), "Pair count mismatch!"
        for op, np in zip(pairs, new_pairs):
            assert op['response']['text'] == np['response']['text'], \
                f"Response mismatch at {op['pair_id']}"
            # Last N original context turns should match
            orig_ctx = [t['text'] for t in op['context']]
            new_ctx = [t['text'] for t in np['context'][-len(orig_ctx):]]
            assert orig_ctx == new_ctx, f"Context mismatch at {op['pair_id']}"

        # Context length stats
        from collections import Counter
        ctx_lens = Counter(len(p['context']) for p in new_pairs)

        filename = f"ground_truth_pairs_ctx{ctx_win}_clean.json"
        output_path = os.path.join(output_dir, filename)

        out_data = {
            "total_pairs": len(new_pairs),
            "videos": data.get("videos", {}),
            "hindi_filter_threshold": data.get("hindi_filter_threshold", 0.5),
            "context_window": ctx_win,
            "pairs": new_pairs
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        print(f"\n  Saved: {filename}")
        print(f"    Pairs: {len(new_pairs)}")
        print(f"    Context turns: {dict(sorted(ctx_lens.items()))}")
        print(f"    All responses match original: YES")
        print(f"    All context suffixes match: YES")

    print(f"\nDONE! Original file unchanged.")


if __name__ == "__main__":
    main()