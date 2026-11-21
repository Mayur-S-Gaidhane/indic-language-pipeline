#!/usr/bin/env python3
"""
mtp3_split_dataset.py — Split Gadkari Interview Dataset (60-20-20)

PURPOSE
-------
Split the 184 ground truth pairs from the Gadkari-Megha Prasad interview
into train/dev/test sets as directed by Prof. Sidharth Ranjan (13 Oct meeting).

Split: 60% train (110), 20% dev (37), 20% test (37) = 184 total.

The split preserves conversation order within each set (pairs are shuffled
for assignment but sorted back by pair index within each set).

OUTPUT
------
  outputs/youtube_eval/splits/
    train.json    (110 pairs)
    dev.json      (37 pairs)
    test.json     (37 pairs)
    split_info.json (metadata)

Student  : Mayur Shriram Gaidhane (M22AIE248)
Supervisor: Prof. Sidharth Ranjan, IIT Jodhpur
"""

import os, sys, json, random
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# Find ground truth
_candidates = [
    os.path.join(ROOT, "ground_truth_data", "ground_truth_pairs_clean.json"),
    os.path.join(ROOT, "ground_truth_pairs_clean.json"),
    os.path.join(ROOT, "ground_truth_data", "ground_truth_pairs.json"),
    os.path.join(ROOT, "ground_truth_pairs.json"),
]
GT_JSON = next((p for p in _candidates if os.path.exists(p)), _candidates[-1])

SPLIT_DIR = os.path.join(ROOT, "outputs", "youtube_eval", "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

SEED = 42  # Reproducible


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split ground truth pairs 60-20-20")
    parser.add_argument("--gt_file", default="",
                        help="Path to ground truth JSON (default: auto-detect)")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output filenames, e.g. '_ctx5' -> train_ctx5.json")
    args = parser.parse_args()

    gt_path = args.gt_file if args.gt_file else GT_JSON
    print(f"Loading: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data["pairs"]
    n = len(pairs)
    print(f"Total pairs: {n}")

    # Compute response length stats (Prof asked for avg length)
    resp_lengths = [len(p["response"]["text"].split()) for p in pairs]
    avg_len = np.mean(resp_lengths)
    median_len = np.median(resp_lengths)
    print(f"Response length — mean: {avg_len:.1f} words, median: {median_len:.0f} words")

    # Shuffle indices, split 60-20-20
    indices = list(range(n))
    random.seed(SEED)
    random.shuffle(indices)

    n_train = int(n * 0.60)   # 110
    n_dev = int(n * 0.20)     # 36 or 37
    # rest goes to test

    train_idx = sorted(indices[:n_train])
    dev_idx = sorted(indices[n_train:n_train + n_dev])
    test_idx = sorted(indices[n_train + n_dev:])

    print(f"\nSplit (seed={SEED}):")
    print(f"  Train: {len(train_idx)} pairs")
    print(f"  Dev:   {len(dev_idx)} pairs")
    print(f"  Test:  {len(test_idx)} pairs")

    train_pairs = [pairs[i] for i in train_idx]
    dev_pairs = [pairs[i] for i in dev_idx]
    test_pairs = [pairs[i] for i in test_idx]

    # Response length stats per split
    for name, split_pairs in [("Train", train_pairs), ("Dev", dev_pairs), ("Test", test_pairs)]:
        lens = [len(p["response"]["text"].split()) for p in split_pairs]
        print(f"  {name} avg response: {np.mean(lens):.1f} words")

    # Save splits
    suffix = args.suffix  # e.g. "_ctx5" or ""
    for name, split_pairs in [("train", train_pairs), ("dev", dev_pairs), ("test", test_pairs)]:
        path = os.path.join(SPLIT_DIR, f"{name}{suffix}.json")
        out = {
            "split": name,
            "n_pairs": len(split_pairs),
            "source": "Gadkari-Megha Prasad ABP Interview",
            "context_window": data.get("context_window", 3),
            "pairs": split_pairs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path}")

    # Save split info
    info = {
        "seed": SEED,
        "total_pairs": n,
        "splits": {
            "train": {"n": len(train_idx), "pair_indices": train_idx},
            "dev": {"n": len(dev_idx), "pair_indices": dev_idx},
            "test": {"n": len(test_idx), "pair_indices": test_idx},
        },
        "avg_response_length_words": round(avg_len, 1),
        "median_response_length_words": int(median_len),
    }
    info_path = os.path.join(SPLIT_DIR, f"split_info{suffix}.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {info_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()